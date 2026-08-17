import sys
from pathlib import Path

root = Path(__file__).parents[1]
sys.path.insert(0, str(root / "api"))

import os
import click
import requests
import asyncio
from websockets.asyncio.client import connect
import json
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
import random
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import numpy as np
from utils import any_format_to_ndarray, save_audio


OUTPUT_FORMAT = "pcm"
SAMPLE_RATE = 16000
SLICE_SECONDS = 0.5
SAVED_ROOT = (root / "test_results").as_posix()
EXAMPLES_PATH = (root / "test/examples.yaml").as_posix()
with open(EXAMPLES_PATH, "r") as f:
    EXAMPLES = YAML(typ="safe").load(f)


async def request_tts(tts_endpoint, text, prompt_id, task_id, is_save=False, saved_dir: str | None = None):
    async def random_segment_text(text, max_len=10):
        print(text)
        while len(text) > 0:
            l = random.randint(1, max_len)
            yield text[:l]
            text = text[l:]

    async with connect(
        tts_endpoint,
        open_timeout=120,
        ping_timeout=120,
        close_timeout=120,
    ) as websocket:
        # send a request
        await websocket.send(
            json.dumps(
                {
                    "req_params": {
                        "prompt_id": prompt_id,
                        "audio_format": OUTPUT_FORMAT,
                        "sample_rate": SAMPLE_RATE,
                        "instruct_text": None,
                        "slice_seconds": SLICE_SECONDS,
                    }
                }
            )
        )

        async def text_sender():
            async for t in random_segment_text(text):
                await websocket.send(json.dumps({"text": t, "done": False}))
                await asyncio.sleep(1e-7)
            await websocket.send(json.dumps({"text": "", "done": True}))

        asyncio.create_task(text_sender())

        all_recv_seconds = []
        whole_audio = []
        root = None
        whole_start = time.perf_counter()
        while True:
            frame_start = time.perf_counter()
            message = await websocket.recv(False)
            message = json.loads(message)

            if message["error"] or message["is_end"]:
                if message["error"]:
                    print(f"{task_id:02} ERROR:", message)

                whole_seconds = time.perf_counter() - whole_start
                all_recv_seconds.append(whole_seconds)
                print(f"{task_id:02} TOTAL RECV:", whole_seconds)
                break

            else:
                frame_seconds = time.perf_counter() - frame_start
                all_recv_seconds.append(frame_seconds)
                print(f"{task_id:02}-{message['index']:02} FRAME RECV:", frame_seconds)
                chunk = any_format_to_ndarray(message["data"], message["audio_format"], message["sample_rate"], SAMPLE_RATE)
                whole_audio.append(chunk)
                if is_save:
                    root = f"{saved_dir}/{task_id}_{prompt_id}_{message['id']}"
                    os.makedirs(f"{root}/chunks", exist_ok=True)
                    save_audio(chunk, f"{root}/chunks/{message['index']}.wav", SAMPLE_RATE)

        whole_audio = np.concatenate(whole_audio)
        duration = len(whole_audio) / SAMPLE_RATE

        if is_save:
            print(f"The audio is saved to {root}/whole.wav.")
            save_audio(whole_audio, f"{root}/whole.wav", SAMPLE_RATE)

        return round(all_recv_seconds[0], 4), round(all_recv_seconds[-1] / duration, 4)


def run_tts(*args):
    return asyncio.run(request_tts(*args))


def eval(tts_endpoint, num_requests, repeat_times, verbose=True):
    if verbose:
        print(f"========== EVAL [{num_requests} REQUESTS]  ==========")

    ttffs = []
    rtfs = []
    for i in range(repeat_times):
        if verbose:
            print(f"========== {i + 1}/{repeat_times} ==========")
        tasks = []
        with ProcessPoolExecutor(max_workers=num_requests) as pool:
            try:
                for j in range(num_requests):
                    tgt_doc = random.choice(EXAMPLES["documents"])
                    tgt_prompt = random.choice(EXAMPLES["prompt_ids"])
                    tasks.append(pool.submit(run_tts, tts_endpoint, tgt_doc, tgt_prompt, i + j))
            except KeyboardInterrupt:
                pool.shutdown()

        for t in tasks:
            ttff, rtf = t.result()
            ttffs.append(ttff)
            rtfs.append(rtf)

    mean_ttff = sum(ttffs) / len(ttffs)
    mean_rtf = sum(rtfs) / len(rtfs)
    if verbose:
        print(f"--> {mean_ttff=} {mean_rtf=}")
    return mean_ttff, mean_rtf


@click.command()
@click.option("--speakers_endpoint", default="http://localhost:12244/speakers")
@click.option("--tts_endpoint", default="ws://localhost:12244/tts")
@click.option("--save_audio", is_flag=True)
@click.option("--eval_ttff", is_flag=True)
@click.option("--warmup_concurrency", type=int, default=1)
@click.option("--warmup_times", type=int, default=1)
@click.option("--eval_max_concurrency", type=int, default=10)
@click.option("--eval_repeat_times", type=int, default=4)
def main(
    speakers_endpoint,
    tts_endpoint,
    save_audio,
    eval_ttff,
    warmup_concurrency,
    warmup_times,
    eval_max_concurrency,
    eval_repeat_times,
):
    # check speakers
    existed_speakers = requests.get(speakers_endpoint).json()
    print(f"Existed Speakers: {existed_speakers}")
    for id in list(EXAMPLES["prompt_ids"]):
        if id not in existed_speakers:
            EXAMPLES["prompt_ids"].remove(id)
    assert len(EXAMPLES["prompt_ids"]) > 0

    if not eval_ttff:
        tgt_doc = random.choice(EXAMPLES["documents"])
        tgt_prompt = random.choice(EXAMPLES["prompt_ids"])
        saved_dir = os.path.join(SAVED_ROOT, "output")
        now_time = str(time.time()).split(".")[0]
        run_tts(tts_endpoint, tgt_doc, tgt_prompt, now_time, save_audio, saved_dir)
        return

    # warm up inference
    print("========== WARM UP ==========")
    eval(tts_endpoint, warmup_concurrency, warmup_times, verbose=False)

    # eval mttff (mean time to first frame) and rtf
    start = time.perf_counter()
    all_results = []
    try:
        for i in range(1, eval_max_concurrency + 1):
            all_results.append((i, *eval(tts_endpoint, i, eval_repeat_times)))
    finally:
        test_time = time.perf_counter() - start
        print(f"========== TOTAL TEST TIME: {test_time} ==========")

        ## save result
        if len(all_results) > 0:
            xs = [i[0] for i in all_results]
            ys_mttff = [i[1] for i in all_results]
            ys_mrtf = [i[2] for i in all_results]

            saved_dir = os.path.join(SAVED_ROOT, "ttff")
            os.makedirs(saved_dir, exist_ok=True)
            fig_path = os.path.join(saved_dir, f"{str(time.time()).split('.')[0]}_{uuid.uuid4().hex[:7]}.png")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            bar1 = ax1.bar(xs, ys_mttff, align="center", color="skyblue")
            ax1.set_xticks(xs)
            ax1.bar_label(bar1)
            ax1.set_title(f"Mean TTFF (test time: {test_time:.2f}s)")
            ax1.set_xlabel("num_requests")
            ax1.set_ylabel("seconds")

            bar2 = ax2.bar(xs, ys_mrtf, align="center", color="salmon")
            ax2.set_xticks(xs)
            ax2.bar_label(bar2)
            ax2.set_title("Mean RTF")
            ax2.set_xlabel("num_requests")
            ax2.set_ylabel("ratio")

            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()


if __name__ == "__main__":
    main()
