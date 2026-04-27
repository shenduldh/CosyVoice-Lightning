import asyncio
import os
import time
import uuid
import http
import traceback
import requests
import numpy as np
from ruamel import yaml
from datetime import datetime
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from io import BytesIO
import soundfile
from loguru import logger

from bases import *
from utils import *
from tts_fast.cosyvoice_fast.entry import CosyVoiceEntry
from tts_fast.cosyvoice_fast.common import CosyVoiceInputType
from seg2stream import SegmentationManager, SegSent2GeneratorConfig, SegSent2StreamConfig, get_phrase_segmenter
from debug import Debugger


class MyApp(FastAPI):
    tts_model: CosyVoiceEntry
    config: dict
    seg_manager: SegmentationManager
    debugger: Debugger
    queues: dict[str, asyncio.Queue] = {}


@asynccontextmanager
async def lifespan(app: MyApp):
    # load config
    config_path = os.environ["CONFIG_PATH"]
    with open(config_path, "r", encoding="utf-8") as f:
        app.config = yaml.YAML().load(f)
    logger.info(f"Successfully load config: {app.config}")

    # load segmentation manager
    match app.config["segmentation"]["mode"]:
        case "bistream":
            bi_cfg = app.config["segmentation"]["bistream"]
            seg_config = SegSent2GeneratorConfig(
                segmentation_suffix=app.config["segmentation"]["seg_suffix"],
                max_waiting_time=bi_cfg["max_waiting_time"],
                max_stream_time=bi_cfg["max_stream_time"],
                first_min_seg_size=bi_cfg["first_min_seg_size"],
                min_seg_size=bi_cfg["min_seg_size"],
            )
        case "unistream":
            uni_cfg = app.config["segmentation"]["unistream"]
            seg_config = SegSent2StreamConfig(
                segmentation_suffix=app.config["segmentation"]["seg_suffix"],
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
    app.seg_manager = SegmentationManager(seg_config, segmenters=[get_phrase_segmenter()])
    app.seg_manager.start()

    async def process_seg_output():
        async for id, text in app.seg_manager.get_async_output():
            if id in app.queues:
                if text is None:
                    app.queues[id].put_nowait(None)
                    app.queues.pop(id)
                elif len(text) > 0:
                    logger.info(f"TTS Segment: {id=} {len(text)=} {text=}")
                    app.queues[id].put_nowait(text)
                    app.debugger.add_text(id, [text])

    seg_output_task = asyncio.create_task(process_seg_output())

    # load tts model
    app.tts_model = CosyVoiceEntry()
    logger.info("TTS model is loaded successfully.")

    # load voice cache
    speaker_cache_path = os.getenv("DEFAULT_SPEAKER_CACHE_PATH", path_to_root("assets", "default_speaker_cache.pt"))
    if os.path.exists(speaker_cache_path):
        speaker_ids = app.tts_model.load_cache(speaker_cache_path)
        logger.info(f"Successfully load speakers: {speaker_ids}")

    # set debug
    app.debugger = Debugger(enabled=bool(int(os.getenv("DEBUG", "0"))))
    logger.info(f"Debug mode: {app.debugger.enabled}")
    app.debugger.patch(app)
    app.debugger.on_startup()

    yield

    app.seg_manager.close()
    await seg_output_task
    app.debugger.on_destroy()


app = MyApp(lifespan=lifespan)


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
    return app.tts_model.get_speakers()


@app.post("/remove")
async def remove_speakers(req: RemoveSpeakersInput) -> dict:
    removed = app.tts_model.remove_speakers(req.prompt_ids)
    return {"removed_speakers": removed}


@app.post("/cache/save")
async def save_cache(req: SaveCacheInput) -> dict:
    cache_path = app.tts_model.save_cache(req.cache_dir, req.filename, req.prompt_ids)
    return {"cache_path": cache_path}


@app.post("/cache/load")
async def load_cache(req: LoadCacheInput) -> dict:
    loaded_speaker_ids = app.tts_model.load_cache(req.cache_path, req.prompt_ids)
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
    if prompt_id in app.tts_model.get_speakers():
        return CloneOutput(existed=True, prompt_id=prompt_id)

    if len(prompt_text.strip()) == 0:
        prompt_text = None

    s = time.time()

    if os.path.exists(prompt_audio):
        app.tts_model.async_request(None, prompt_audio, prompt_text, None, prompt_id, loudness)
    elif prompt_audio.startswith("http"):
        prompt_audio = requests.get(prompt_audio).content
        with NamedTemporaryFile() as f:
            f.write(prompt_audio)
            f.flush()
            app.tts_model.async_request(None, f.name, prompt_text, None, prompt_id, loudness)
    else:
        prompt_audio = any_format_to_ndarray(prompt_audio, audio_format, sample_rate)
        with NamedTemporaryFile(suffix=".wav") as f:
            save_audio(prompt_audio, f.name, 16000)
            f.flush()
            app.tts_model.async_request(None, f.name, prompt_text, None, prompt_id, loudness)

    e = time.time()
    logger.info(f"Clone time: {e - s}")

    return CloneOutput(existed=False, prompt_id=prompt_id)


@app.post("/tts")
async def tts(req: TTSInput):
    logger.info("Request Params: %s" % req)

    prompt_id = req.prompt_id
    instruct_text = req.instruct_text

    if prompt_id not in app.tts_model.get_speakers():
        return JSONResponse("No such speaker.", http.HTTPStatus.NOT_FOUND)

    if instruct_text is not None and len(instruct_text) == 0:
        instruct_text = None

    audio_ndarray = []
    async for chunk in app.tts_model.async_request(
        req.text,
        None,
        None,
        instruct_text,
        prompt_id,
        split_text=True,
        stream=True,
        generation_params=req.generation_params.model_dump(exclude_unset=True),
    ):
        audio_ndarray.append(chunk)
    audio_ndarray = np.concatenate(audio_ndarray)

    if req.return_base64:
        return TTSOutput(
            audio=format_ndarray_to_base64(
                audio_ndarray,
                app.tts_model.sample_rate,
                req.sample_rate,
                req.audio_format,
            )
        )
    else:
        buffer = BytesIO()
        soundfile.write(buffer, audio_ndarray, format="wav", samplerate=app.tts_model.sample_rate)
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"},
        )


@dataclass
class TTSTask:
    params: TTSStreamRequestParameters
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    date: datetime = field(default_factory=datetime.now)
    counter: int = 0
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


async def run_task(tts_task: TTSTask, websocket: WebSocket):
    # from pyinstrument import Profiler
    # profiler = Profiler(interval=0.0001, async_mode="enabled", use_timing_thread=True)
    # profiler.start()

    # from torch.profiler import profile, ProfilerActivity
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:

    try:
        output_stream = app.tts_model.async_request(
            tts_task.queue,
            None,
            None,
            tts_task.params.instruct_text,
            tts_task.params.prompt_id,
            stream=True,
            generation_params=tts_task.params.generation_params.model_dump(exclude_unset=True),
            input_type=CosyVoiceInputType.QUEUE,
        )
        repacking_size = int(app.tts_model.sample_rate * tts_task.params.slice_seconds)
        capacity = int(app.tts_model.sample_rate * 10)
        repacked_stream = async_repack(output_stream, repacking_size, repacking_size, capacity)

        async for chunk_ndarray in repacked_stream:
            if app.config["tts"]["do_removing_silence"]:
                chunk_ndarray = remove_silence(
                    chunk_ndarray,
                    app.tts_model.sample_rate,
                    (
                        app.config["tts"]["first_left_retention_seconds"]
                        if tts_task.counter == 0
                        else app.config["tts"]["left_retention_seconds"]
                    ),
                    app.config["tts"]["right_retention_seconds"],
                )
            chunk_base64 = format_ndarray_to_base64(
                chunk_ndarray,
                app.tts_model.sample_rate,
                tts_task.params.sample_rate,
                tts_task.params.audio_format,
            )
            await websocket.send_json(
                TTSStreamOutput(
                    id=tts_task.id,
                    is_end=False,
                    index=tts_task.counter,
                    data=chunk_base64,
                    audio_format=tts_task.params.audio_format,
                    sample_rate=tts_task.params.sample_rate,
                ).model_dump()
            )
            tts_task.counter += 1
            app.debugger.add_chunk(tts_task.id, chunk_ndarray)

        await websocket.send_json(TTSStreamOutput(id=tts_task.id, is_end=True, index=tts_task.counter).model_dump())
        app.debugger.save(tts_task.id, tts_task, app.tts_model.sample_rate)
    except BaseException as e:
        asyncio.create_task(output_stream.athrow(StopAsyncIteration))
        if not isinstance(e, asyncio.CancelledError):
            raise

    # prof.export_chrome_trace("torch_profile.json")
    # profiler.stop()
    # profiler.print()
    # with open("profile.html", "w") as f:
    #     f.write(profiler.output_html())


@app.websocket("/tts")
async def tts(websocket: WebSocket):
    await websocket.accept()

    curr_task = None
    running_task = None

    while True:
        try:
            req = await websocket.receive_json()
            if running_task is None:
                req = TTSStreamRequestInput(**req)
                req_params = req.req_params
                if req_params.prompt_id not in app.tts_model.get_speakers():
                    raise ValueError("No such speaker.")
                if req_params.instruct_text is not None and len(req_params.instruct_text) == 0:
                    req_params.instruct_text = None

                curr_task = TTSTask(params=req_params)
                logger.info(f"TTS Request: {curr_task}")
                app.queues[curr_task.id] = curr_task.queue
                running_task = asyncio.create_task(run_task(curr_task, websocket))
            else:
                req = TTSStreamTextInput(**req)
                logger.info(f"TTS Stream: {req}")
                app.seg_manager.add_text(curr_task.id, req.text)
                if req.done:
                    app.seg_manager.add_text(curr_task.id, None)
                    await running_task
                    curr_task = None
                    running_task = None

        except BaseException as e:
            if curr_task is not None:
                if curr_task.id in app.queues:
                    app.seg_manager.add_text(curr_task.id, None)
                curr_task = None
            if running_task is not None:
                if not running_task.done():
                    running_task.cancel()
                    await running_task
                running_task = None
            if isinstance(e, WebSocketDisconnect):
                break
            logger.error(f"Error in websocket:\n{traceback.format_exc()}")
            await websocket.send_json({"error": True, "message": whats_wrong_with(e)})
