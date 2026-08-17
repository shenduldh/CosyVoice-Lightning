import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "api"))

import os
import requests
import click
import librosa
from utils import ndarray_to_base64_no_header


@click.command()
@click.option("--ip")
@click.option("--port")
@click.option("--speakers_path")
@click.option("--loudness", default=20.0)
@click.option("--base64", is_flag=True)
def main(ip, port, speakers_path, loudness, base64):
    audio_formats = ["wav", "mp3"]

    for speaker_id in os.listdir(speakers_path):
        audio_paths = list(
            filter(
                os.path.exists,
                [os.path.join(speakers_path, speaker_id, f"audio.{fmt}") for fmt in audio_formats],
            )
        )
        transcript_path = os.path.join(speakers_path, speaker_id, "transcript.txt")

        if len(audio_paths) > 0 and os.path.exists(transcript_path):
            audio_path = audio_paths[0]
            if base64:
                audio_ndarray, _ = librosa.load(audio_path, sr=16000)
                audio = ndarray_to_base64_no_header(audio_ndarray)
            else:
                audio = audio_path

            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                res = requests.post(
                    f"http://{ip}:{port}/clone",
                    json={
                        "prompt_text": transcript,
                        "prompt_audio": audio,
                        "prompt_id": speaker_id,
                        "audio_format": "pcm",
                        "sample_rate": 16000,
                        "loudness": loudness,
                    },
                )
                print(res.json())


if __name__ == "__main__":
    main()
