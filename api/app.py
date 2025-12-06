import sys
import asyncio
import uvloop
from utils import path_to_root
from logger import logger

sys.path.insert(0, path_to_root())
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import os
import time
import uuid
import http
import traceback
import requests
import numpy as np
from ruamel import yaml
from typing import Union
from datetime import datetime
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from io import BytesIO
import soundfile

from bases import *
from utils import *
from debug import Debug, debug as debug_instance
from tts_fast.pipeline import CosyVoice2Pipeline, CosyVoiceInputType
from seg2stream import (
    SegmentationManager,
    SegSent2GeneratorConfig,
    SegSent2StreamConfig,
)


class Global:
    tts_model: Union[None | CosyVoice2Pipeline] = None
    config: Union[None | dict] = None
    seg_manager: Union[None | SegmentationManager] = None
    queues: dict[str, asyncio.Queue] = {}
    debug: Debug = debug_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load config
    config_path = os.environ["CONFIG_PATH"]
    with open(config_path, "r", encoding="utf-8") as f:
        Global.config = yaml.YAML().load(f)
    logger.info(f"Successfully load config: {Global.config}")

    # load segmentation manager
    match Global.config["segmentation"]["mode"]:
        case "bistream":
            bi_cfg = Global.config["segmentation"]["bistream"]
            seg_config = SegSent2GeneratorConfig(
                segmentation_suffix=Global.config["segmentation"]["seg_suffix"],
                max_waiting_time=bi_cfg["max_waiting_time"],
                max_stream_time=bi_cfg["max_stream_time"],
                first_min_seg_size=bi_cfg["first_min_seg_size"],
                min_seg_size=bi_cfg["min_seg_size"],
            )
        case "unistream":
            uni_cfg = Global.config["segmentation"]["unistream"]
            seg_config = SegSent2StreamConfig(
                segmentation_suffix=Global.config["segmentation"]["seg_suffix"],
                first_max_accu_time=uni_cfg["first_max_accu_time"],
                max_accu_time=uni_cfg["max_accu_time"],
                first_max_buffer_size=uni_cfg["first_max_buffer_size"],
                max_buffer_size=uni_cfg["max_buffer_size"],
                max_waiting_time=uni_cfg["max_waiting_time"],
                max_stream_time=uni_cfg["max_stream_time"],
                first_min_seg_size=uni_cfg["first_min_seg_size"],
                min_seg_size=uni_cfg["min_seg_size"],
                max_seg_size=uni_cfg["max_seg_size"],
                loose_steps=uni_cfg["loose_steps"],
                loose_size=uni_cfg["loose_size"],
                fade_in_out_time=uni_cfg["fade_in_out_time"],
                seconds_per_word=uni_cfg["seconds_per_word"],
            )
    Global.seg_manager = SegmentationManager(seg_config)
    Global.seg_manager.start()

    async def process_seg_output():
        async for id, text in Global.seg_manager.get_async_output():
            if id in Global.queues:
                Global.queues[id].put_nowait(text)
                logger.info(f"TTS Segment: id={id} text={text}")
                if text is None:
                    Global.queues.pop(id)
                else:
                    Global.debug.add_text(id, [text])

    seg_output_task = asyncio.create_task(process_seg_output())

    # load tts model
    tts_model_dir = os.environ["TTS_MODEL_DIR"]
    Global.tts_model = CosyVoice2Pipeline(tts_model_dir)
    logger.info("TTS model is loaded successfully.")

    # load voice cache
    speaker_cache_path = os.getenv(
        "DEFAULT_SPEAKER_CACHE_PATH", path_to_root("assets", "default_speaker_cache.pt")
    )
    if os.path.exists(speaker_cache_path):
        speaker_ids = Global.tts_model.load_cache(speaker_cache_path)
        logger.info(f"Successfully load speakers: {speaker_ids}")

    # set debug
    Global.debug.set_enabled(os.getenv("DEBUG", "0") == "1")
    logger.info(f"Debug mode: {Global.debug.enabled}")
    Global.debug.patch(app)
    Global.debug.on_startup()

    yield

    Global.seg_manager.close()
    await seg_output_task
    Global.debug.on_destroy()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def general_exception_handler(request, e: Exception):
    logger.error(f"Error in Response: {e}")
    return JSONResponse(str(e), http.HTTPStatus.INTERNAL_SERVER_ERROR)


@app.get("/")
async def index() -> str:
    return f"Hello."


@app.get("/alive")
async def alive() -> dict:
    return {"status": "alive"}


@app.get("/speakers")
async def get_speakers() -> list:
    return Global.tts_model.get_speakers()


@app.post("/remove")
async def remove_speakers(req: RemoveSpeakersInput) -> dict:
    removed = Global.tts_model.remove_speakers(req.prompt_ids)
    return {"removed_speakers": removed}


@app.post("/cache/save")
async def save_cache(req: SaveCacheInput) -> dict:
    cache_path = Global.tts_model.save_cache(
        req.cache_dir, req.filename, req.prompt_ids
    )
    return {"cache_path": cache_path}


@app.post("/cache/load")
async def load_cache(req: LoadCacheInput) -> dict:
    loaded_speaker_ids = Global.tts_model.load_cache(req.cache_path, req.prompt_ids)
    return {"loaded_speakers": loaded_speaker_ids}


@app.post("/clone")
async def clone(req: CloneInput) -> CloneOutput:
    logger.info("Request Params: %s" % truncate_long_str(req.model_dump()))

    prompt_id = req.prompt_id
    prompt_text = req.prompt_text
    prompt_audio = req.prompt_audio
    loudness = float(req.loudness)
    sample_rate = req.sample_rate
    audio_format = req.audio_format

    if prompt_id is None:
        prompt_id = f"{uuid.uuid4().hex[:7]}_{datetime.now().strftime('%Y-%m-%d')}"
    if prompt_id in Global.tts_model.get_speakers():
        return CloneOutput(existed=True, prompt_id=prompt_id)

    if len(prompt_text.strip()) == 0:
        prompt_text = None

    s = time.time()
    test_text = "这是一段测试文本。"

    if os.path.exists(prompt_audio):
        async for _ in Global.tts_model.async_generate(
            None, test_text, prompt_text, prompt_audio, None, prompt_id, loudness
        ):
            pass
    elif prompt_audio.startswith("http"):
        prompt_audio = requests.get(prompt_audio).content
        with NamedTemporaryFile() as f:
            f.write(prompt_audio)
            f.flush()
            async for _ in Global.tts_model.async_generate(
                None, test_text, prompt_text, f.name, None, prompt_id, loudness
            ):
                pass
    else:
        prompt_audio = any_format_to_ndarray(prompt_audio, audio_format, sample_rate)
        with NamedTemporaryFile(suffix=".wav") as f:
            save_audio(prompt_audio, f.name, 16000)
            f.flush()
            async for _ in Global.tts_model.async_generate(
                None, test_text, prompt_text, f.name, None, prompt_id, loudness
            ):
                pass

    e = time.time()
    logger.info(f"Clone time: {e - s}")

    return CloneOutput(existed=False, prompt_id=prompt_id)


@app.post("/tts")
async def tts(req: TTSInput) -> TTSOutput:
    logger.info("Request Params: %s" % req)

    prompt_id = req.prompt_id
    instruct_text = req.instruct_text

    if prompt_id not in Global.tts_model.get_speakers():
        return JSONResponse("No such speaker.", http.HTTPStatus.NOT_FOUND)

    if instruct_text is not None and len(instruct_text) == 0:
        instruct_text = None

    audio_ndarray = []
    async for chunk in Global.tts_model.async_generate(
        None, req.text, None, None, instruct_text, prompt_id, split_text=True
    ):
        audio_ndarray.append(chunk)
    audio_ndarray = np.concatenate(audio_ndarray)

    if req.return_base64:
        return TTSOutput(
            audio=format_ndarray_to_base64(
                audio_ndarray,
                Global.tts_model.sample_rate,
                req.sample_rate,
                req.audio_format,
            )
        )
    else:
        buffer = BytesIO()
        soundfile.write(
            buffer, audio_ndarray, format="wav", samplerate=Global.tts_model.sample_rate
        )
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"},
        )


@dataclass
class TTSInfo:
    request: TTSStreamRequestParameters
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    date: datetime = field(default_factory=datetime.now)
    count: int = 0
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


async def tts_job(tts_info: TTSInfo, websocket: WebSocket):
    try:
        output_stream = Global.tts_model.async_generate(
            tts_info.id,
            tts_info.queue,
            None,
            None,
            tts_info.request.instruct_text,
            tts_info.request.prompt_id,
            stream=True,
            input_type=CosyVoiceInputType.QUEUE,
        )
        min_repacking_size = Global.tts_model.sample_rate * 0.01

        async for chunk_ndarray in async_repack(output_stream, min_repacking_size):
            if Global.config["tts"]["do_removing_silence"]:
                chunk_ndarray = remove_silence(
                    chunk_ndarray,
                    Global.tts_model.sample_rate,
                    (
                        Global.config["tts"]["first_left_retention_seconds"]
                        if tts_info.count == 0
                        else Global.config["tts"]["left_retention_seconds"]
                    ),
                    Global.config["tts"]["right_retention_seconds"],
                )
            chunk_base64 = format_ndarray_to_base64(
                chunk_ndarray,
                Global.tts_model.sample_rate,
                tts_info.request.sample_rate,
                tts_info.request.audio_format,
            )
            await websocket.send_json(
                TTSStreamOutput(
                    id=tts_info.id,
                    is_end=False,
                    index=tts_info.count,
                    data=chunk_base64,
                    audio_format=tts_info.request.audio_format,
                    sample_rate=tts_info.request.sample_rate,
                ).model_dump()
            )
            tts_info.count += 1
            Global.debug.add_chunk(tts_info.id, chunk_ndarray)

        await websocket.send_json(
            TTSStreamOutput(
                id=tts_info.id, is_end=True, index=tts_info.count
            ).model_dump()
        )
        Global.debug.save(tts_info.id, tts_info, Global.tts_model.sample_rate)
    except BaseException as e:
        asyncio.create_task(output_stream.athrow(StopAsyncIteration))
        if not isinstance(e, asyncio.CancelledError):
            raise


@app.websocket("/tts")
async def tts(websocket: WebSocket):
    await websocket.accept()

    tts_info = None
    running_job = None

    while True:
        try:
            req = await websocket.receive_json()
            if running_job is None:
                req = TTSStreamRequestInput(**req)
                req_params = req.req_params
                if req_params.prompt_id not in Global.tts_model.get_speakers():
                    raise ValueError("No such speaker.")
                if (
                    req_params.instruct_text is not None
                    and len(req_params.instruct_text) == 0
                ):
                    req_params.instruct_text = None

                tts_info = TTSInfo(request=req_params)
                logger.info(f"TTS Request: {tts_info}")
                Global.queues[tts_info.id] = tts_info.queue
                running_job = asyncio.create_task(tts_job(tts_info, websocket))
            else:
                req = TTSStreamTextInput(**req)
                logger.info("TTS Stream: %s" % req)
                if len(req.text) > 0:
                    Global.seg_manager.add_text(tts_info.id, req.text)
                if req.done:
                    Global.seg_manager.add_text(tts_info.id, None)
                    await running_job
                    tts_info = None
                    running_job = None

        except BaseException as e:
            if tts_info is not None:
                if tts_info.id in Global.queues:
                    Global.seg_manager.add_text(tts_info.id, None)
                tts_info = None
            if running_job is not None:
                if not running_job.done():
                    running_job.cancel()
                    await running_job
                running_job = None
            if isinstance(e, WebSocketDisconnect):
                break
            logger.error(f"Error in websocket: {traceback.format_exc()}")
            await websocket.send_json({"error": True, "message": whats_wrong_with(e)})
