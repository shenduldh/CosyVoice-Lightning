import os
import tqdm
from collections import defaultdict
import json
import onnxruntime
import whisper
import numpy as np
import torch
import pandas as pd
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import ray
from ray.util import ActorPool
import click


def search_and_organize_utts(
    *data_dirs,
    output_dir,
    audio_suffix=".wav",
    text_suffix=".txt",
    apply_dpo=False,
    reject_audio_suffix=".reject.wav",
):
    """
    Search all utterances within the specified directory
    and organize them into utt2audio, utt2text, utt2spk, spk2utt mappings.
    Note the portion to the left of the "_" in the utterance name
    indicates the speaker's name.
    """
    text_paths = []
    audio_paths = []
    reject_audio_paths = []
    for d in data_dirs:
        for root, _, files in tqdm.tqdm(os.walk(d), desc=f"Searching {d}"):
            for f in files:
                if f.endswith(text_suffix):
                    text_paths.append(os.path.join(root, f))
                elif f.endswith(reject_audio_suffix):
                    reject_audio_paths.append(os.path.join(root, f))
                elif f.endswith(audio_suffix):
                    audio_paths.append(os.path.join(root, f))

    matched = []
    for audio_path in tqdm.tqdm(audio_paths, desc="Matching"):
        prefix = audio_path.rstrip(audio_suffix)
        text_path = prefix + text_suffix
        reject_audio_path = prefix + reject_audio_suffix

        this_matched = [audio_path, None, None]

        if text_path in text_paths:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().replace("\n", "").strip()
                this_matched[1] = text
        if reject_audio_path in reject_audio_paths:
            this_matched[2] = reject_audio_path

        if (this_matched[1] is not None) and (
            (not apply_dpo) or (apply_dpo and this_matched[2] is not None)
        ):
            matched.append(this_matched)

    utt2any = {}
    spk2utt = defaultdict(lambda: [])
    for audio_path, text, reject_audio_path in matched:
        utt_name = os.path.basename(audio_path).rstrip(audio_suffix)
        spk_name = utt_name.split("_")[0]
        utt2any[utt_name] = {
            "audio": audio_path,
            "text": text,
            "reject_audio": reject_audio_path,
            "speaker": spk_name,
        }
        spk2utt[spk_name].append(utt_name)

    os.makedirs(output_dir, exist_ok=True)

    utt2any_path = os.path.join(output_dir, "utt2any.json")
    spk2utt_path = os.path.join(output_dir, "spk2utt.json")
    with open(utt2any_path, "w", encoding="utf-8") as f:
        json.dump(utt2any, f, ensure_ascii=False)
    with open(spk2utt_path, "w", encoding="utf-8") as f:
        json.dump(spk2utt, f, ensure_ascii=False)

    return utt2any_path, spk2utt_path


class Extractor:
    """
    The extractor is used to extract speech tokens and utterance embeddings
    by the Cam++ model and the speech tokenizer.
    """

    def __init__(self, campplus_onnx_path, speech_tokenizer_onnx_path):
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.intra_op_num_threads = 1

        self.campplus_session = onnxruntime.InferenceSession(
            campplus_onnx_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_onnx_path,
            sess_options=options,
            providers=["CUDAExecutionProvider"],
        )

    def extract_embedding(self, utt_name, audio_path, flag):
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(audio)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        return flag, utt_name, embedding

    def extract_speech_token(self, utt_name, audio_path, flag):
        audio, sample_rate = torchaudio.load(audio_path, backend="soundfile")
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # not support extract speech token for audio longer than 30s
        if audio.shape[1] / 16000 > 30:
            speech_token = []
        else:
            feat = whisper.log_mel_spectrogram(audio, n_mels=128)
            speech_token = (
                self.speech_tokenizer_session.run(
                    None,
                    {
                        self.speech_tokenizer_session.get_inputs()[
                            0
                        ].name: feat.detach().cpu().numpy(),
                        self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                            [feat.shape[2]], dtype=np.int32
                        ),
                    },
                )[0]
                .flatten()
                .tolist()
            )
        return flag, utt_name, speech_token


def extract_features(
    utt2any_path,
    campplus_onnx_path,
    speech_tokenizer_onnx_path,
    output_dir,
    num_cpus=0.2,
    num_gpus=0.2,
    num_actors=10,
):
    """
    Bacth extract the speech tokens and embeddings for all utterances
    along the specified path.
    """
    with open(utt2any_path, "r", encoding="utf-8") as f:
        utt2any = json.load(f)

    actor = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(Extractor)
    pool = ActorPool(
        [
            actor.remote(campplus_onnx_path, speech_tokenizer_onnx_path)
            for _ in range(num_actors)
        ]
    )

    token_task_count = 0
    embed_task_count = 0
    for utt_name, val in utt2any.items():
        audio_path = val["audio"]
        pool.submit(
            lambda actor, args: actor.extract_embedding.remote(*args),
            (utt_name, audio_path, "embedding"),
        )
        embed_task_count += 1
        pool.submit(
            lambda actor, args: actor.extract_speech_token.remote(*args),
            (utt_name, audio_path, "token"),
        )
        token_task_count += 1
        reject_audio_path = val["reject_audio"]
        if reject_audio_path is not None:
            pool.submit(
                lambda actor, args: actor.extract_speech_token.remote(*args),
                (utt_name, reject_audio_path, "reject_token"),
            )
            token_task_count += 1

    utt2feat = defaultdict(dict)
    spk2embedding = {}
    with (
        tqdm.tqdm(total=embed_task_count, desc="Extracting embeddings") as embed_pbar,
        tqdm.tqdm(total=token_task_count, desc="Extracting tokens") as token_pbar,
    ):
        while pool.has_next():
            flag, utt_name, feat = pool.get_next()
            if flag == "embedding":
                utt2feat[utt_name]["embedding"] = feat
                spk = utt2any[utt_name]["speaker"]
                if spk not in spk2embedding:
                    spk2embedding[spk] = []
                spk2embedding[spk].append(feat)
                embed_pbar.update(1)
            elif flag == "token":
                utt2feat[utt_name]["token"] = feat
                token_pbar.update(1)
            elif flag == "reject_token":
                utt2feat[utt_name]["reject_token"] = feat
                token_pbar.update(1)

    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    os.makedirs(output_dir, exist_ok=True)
    utt2feat_path = os.path.join(output_dir, "utt2feat.pt")
    spk2embedding_path = os.path.join(output_dir, "spk2embedding.pt")
    torch.save(utt2feat, utt2feat_path)
    torch.save(spk2embedding, spk2embedding_path)

    return utt2feat_path, spk2embedding_path


def make_parquets(
    utt2any_path,
    utt2feat_path,
    spk2embedding_path,
    parquet_dir,
    num_utts_per_parquet=1000,
    num_cpus=0.1,
):
    """
    Make all the outputs into parquets.
    """
    with open(utt2any_path, "r", encoding="utf-8") as f:
        utt2any = json.load(f)
    utt2feat = torch.load(utt2feat_path)
    spk2embedding = torch.load(spk2embedding_path)

    shared = ray.put(
        {"utt2any": utt2any, "utt2feat": utt2feat, "spk2embedding": spk2embedding}
    )

    @ray.remote(num_cpus=num_cpus)
    def job(utts, parquet_path, shared):
        utt2any = shared["utt2any"]
        utt2feat = shared["utt2feat"]
        spk2embedding = shared["spk2embedding"]

        audio_bytes_list = []
        audio_path_list = []
        text_list = []
        spk_list = []
        utt_embedding_list = []
        spk_embedding_list = []
        speech_token_list = []
        reject_speech_token_list = []
        for utt_name in utts:
            audio_path = utt2any[utt_name]["audio"]
            spk_name = utt2any[utt_name]["speaker"]
            with open(audio_path, "rb") as f:
                audio_bytes_list.append(f.read())
            audio_path_list.append(audio_path)
            text_list.append(utt2any[utt_name]["text"])
            spk_list.append(spk_name)
            utt_embedding_list.append(utt2feat[utt_name]["embedding"])
            spk_embedding_list.append(spk2embedding[spk_name])
            speech_token_list.append(utt2feat[utt_name]["token"])
            if "reject_token" in utt2feat[utt_name]:
                reject_speech_token_list.append(utt2feat[utt_name]["reject_token"])

        df = pd.DataFrame()
        df["utt"] = utts
        df["wav"] = audio_path_list
        df["audio_data"] = audio_bytes_list
        df["text"] = text_list
        df["spk"] = spk_list
        df["utt_embedding"] = utt_embedding_list
        df["spk_embedding"] = spk_embedding_list
        df["speech_token"] = speech_token_list
        if len(reject_speech_token_list) > 0:
            df["reject_speech_token"] = reject_speech_token_list
        df.to_parquet(parquet_path)

    parquet_dir = os.path.join(parquet_dir, "parquet")
    os.makedirs(parquet_dir, exist_ok=True)

    utts = list(utt2any.keys())
    utt_chunks = [
        utts[i : i + num_utts_per_parquet]
        for i in range(0, len(utts), num_utts_per_parquet)
    ]
    tasks = []
    parquet_list = []
    with tqdm.tqdm(total=len(utt_chunks), desc="Making parquets") as pbar:
        for i, this_utts in enumerate(utt_chunks):
            parquet_path = os.path.join(parquet_dir, "{:09d}.tar".format(i))
            tasks.append(job.remote(this_utts, parquet_path, shared))
            parquet_list.append(parquet_path)
        for task in tasks:
            ray.get(task)
            pbar.update(1)

    parquet_path = os.path.join(parquet_dir, "parquet_list")
    with open(parquet_path, "w", encoding="utf-8") as f:
        for name in parquet_list:
            f.write(name + "\n")
    return parquet_path


@click.command()
@click.option("--data_dirs", multiple=True, help="Paths of the data directories.")
@click.option("--output_dir", help="Path of the output directory.")
@click.option("--campplus", help="Path of the campplus onnx model.")
@click.option("--speech_tokenizer", help="Path of the speech tokenizer onnx model.")
@click.option("--audio_suffix", default=".wav")
@click.option("--text_suffix", default=".txt")
@click.option("--reject_audio_suffix", default=".reject.wav")
@click.option("--apply_dpo", default=False)
@click.option("--num_cpus", default=os.cpu_count())
@click.option("--num_gpus", default=2)
@click.option("--num_workers", default=20)
def main(
    data_dirs,
    output_dir,
    campplus,
    speech_tokenizer,
    audio_suffix,
    text_suffix,
    reject_audio_suffix,
    apply_dpo,
    num_cpus,
    num_gpus,
    num_workers,
):
    campplus_onnx_path = campplus
    speech_tokenizer_onnx_path = speech_tokenizer

    utt2any_path, spk2utt_path = search_and_organize_utts(
        *data_dirs,
        output_dir=output_dir,
        audio_suffix=audio_suffix,
        text_suffix=text_suffix,
        apply_dpo=apply_dpo,
        reject_audio_suffix=reject_audio_suffix,
    )
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    utt2feat_path, spk2embedding_path = extract_features(
        utt2any_path,
        campplus_onnx_path,
        speech_tokenizer_onnx_path,
        output_dir,
        num_cpus=num_cpus / num_workers,
        num_gpus=(num_gpus - 0.2) / num_workers,
        num_actors=num_workers,
    )
    parquet_path = make_parquets(
        utt2any_path,
        utt2feat_path=utt2feat_path,
        spk2embedding_path=spk2embedding_path,
        parquet_dir=output_dir,
    )

    print("Preparing data is finished. Data is saved into these paths:")
    print("utt2any_path:", utt2any_path)
    print("spk2utt_path:", spk2utt_path)
    print("utt2feat_path:", utt2feat_path)
    print("spk2embedding_path:", spk2embedding_path)
    print("parquet_path:", parquet_path)


if __name__ == "__main__":
    main()
