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
from contextlib import aclosing
from loguru import logger

from bases import *
from utils import (
    truncate_long_str,
    any_format_to_ndarray,
    save_audio,
    async_repack,
    remove_silence,
    whats_wrong_with,
    format_ndarray_to_base64,
    get_av_audio_encoder,
)
from debug import Debugger
from tts_fast.cosyvoice_fast.entry import CosyVoiceEntry
from tts_fast.cosyvoice_fast.common import CosyVoiceInputType
from seg2stream import SegmentationManager, SegSent2GeneratorConfig, SegSent2StreamConfig, get_phrase_segmenter


@dataclass
class TTSTask:
    params: TTSStreamRequestParameters
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    date: datetime = field(default_factory=datetime.now)
    counter: int = 0
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    accum_len: int = 0


class MyApp(FastAPI):
    tts_model: CosyVoiceEntry
    config: dict
    seg_manager: SegmentationManager
    debugger: Debugger
    tasks: dict[str, TTSTask] = {}


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
            if id in app.tasks:
                this_task = app.tasks[id]
                if text is None:
                    this_task.queue.put_nowait(None)
                    app.tasks.pop(id)
                elif len(text) > 0:
                    this_len = len(text)
                    max_len = this_task.params.max_text_length
                    this_task.accum_len += this_len
                    if this_task.accum_len > max_len:
                        logger.info(f"Exceed Limit: {id=} {max_len=} accum_len={this_task.accum_len}")
                        this_task.queue.put_nowait(None)
                        app.tasks.pop(id)
                    else:
                        logger.info(f"TTS Segment: {id=} {this_len=} accum_len={this_task.accum_len} {text=}")
                        app.tasks[id].queue.put_nowait(text)
                        app.debugger.add_text(id, [text])

    seg_output_task = asyncio.create_task(process_seg_output())

    # load tts model
    app.tts_model = CosyVoiceEntry()
    logger.info("TTS model is loaded successfully.")

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
    return "Hello."


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
    logger.info(f"Request Params: {truncate_long_str(req.model_dump())}")

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
        app.tts_model.async_request(None, None, prompt_audio, prompt_text, None, prompt_id, loudness)
    elif prompt_audio.startswith("http"):
        prompt_audio = requests.get(prompt_audio).content
        with NamedTemporaryFile() as f:
            f.write(prompt_audio)
            f.flush()
            app.tts_model.async_request(None, None, f.name, prompt_text, None, prompt_id, loudness)
    else:
        prompt_audio = any_format_to_ndarray(prompt_audio, audio_format, sample_rate)
        with NamedTemporaryFile(suffix=".wav") as f:
            save_audio(prompt_audio, f.name, 16000)
            f.flush()
            app.tts_model.async_request(None, None, f.name, prompt_text, None, prompt_id, loudness)

    e = time.time()
    logger.info(f"Clone time: {e - s}")

    return CloneOutput(existed=False, prompt_id=prompt_id)


@app.post("/tts")
async def tts(req: TTSInput):
    logger.info(f"Request Params: {req}")

    prompt_id = req.prompt_id
    instruct_text = req.instruct_text

    if prompt_id not in app.tts_model.get_speakers():
        return JSONResponse("No such speaker.", http.HTTPStatus.NOT_FOUND)

    if instruct_text is not None and len(instruct_text) == 0:
        instruct_text = None

    audio_ndarray = []
    async for chunk in app.tts_model.async_request(
        tts_text=req.text,
        prompt_audio=None,
        prompt_text=None,
        instruct_text=instruct_text,
        speaker_id=prompt_id,
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


async def run_task(tts_task: TTSTask, websocket: WebSocket):
    try:
        start_time = time.perf_counter()

        output_stream = app.tts_model.async_request(
            tts_task.id,
            tts_task.queue,
            None,
            None,
            tts_task.params.instruct_text,
            tts_task.params.prompt_id,
            stream=True,
            generation_params=tts_task.params.generation_params.model_dump(exclude_unset=True),
            input_type=CosyVoiceInputType.QUEUE,
        )
        audio_encoder = get_av_audio_encoder(app.tts_model.sample_rate, tts_task.params.sample_rate, tts_task.params.audio_format)
        repacking_size = int(app.tts_model.sample_rate * tts_task.params.slice_seconds)
        capacity = int(app.tts_model.sample_rate * 10)
        repacked_stream = async_repack(output_stream, repacking_size, repacking_size, capacity)

        async with aclosing(output_stream), aclosing(repacked_stream) as stream:
            async for chunk_ndarray in stream:
                if app.config["tts"]["do_removing_silence"]:
                    chunk_ndarray = remove_silence(
                        chunk_ndarray,
                        app.tts_model.sample_rate,
                        (app.config["tts"]["first_left_retention_seconds"] if tts_task.counter == 0 else app.config["tts"]["left_retention_seconds"]),
                        app.config["tts"]["right_retention_seconds"],
                    )
                chunk_base64 = await asyncio.to_thread(audio_encoder.encode, chunk_ndarray)
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
                if tts_task.counter == 0:
                    spent_time = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.info(f"TTS first response: id={tts_task.id} spent_time={spent_time}")
                tts_task.counter += 1
                app.debugger.add_chunk(tts_task.id, chunk_ndarray)

        await websocket.send_json(TTSStreamOutput(id=tts_task.id, is_end=True, index=tts_task.counter).model_dump())
        app.debugger.save(tts_task.id, tts_task, app.tts_model.sample_rate)

    except BaseException as e:
        if not isinstance(e, asyncio.CancelledError):
            raise

    finally:
        app.debugger.discard(tts_task.id)
        try:
            audio_encoder.close()
        except:
            pass


@app.websocket("/tts")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()

    curr_task = None
    running_task = None

    while True:
        try:
            req = await websocket.receive_json()
            if running_task is None or curr_task is None:
                req = TTSStreamRequestInput(**req)
                req_params = req.req_params
                if req_params.prompt_id not in app.tts_model.get_speakers():
                    raise ValueError("No such speaker.")
                if req_params.instruct_text is not None and len(req_params.instruct_text) == 0:
                    req_params.instruct_text = None

                curr_task = TTSTask(params=req_params)
                logger.info(f"TTS Request: {curr_task}")
                app.tasks[curr_task.id] = curr_task
                running_task = asyncio.create_task(run_task(curr_task, websocket))
            else:
                req = TTSStreamTextInput(**req)
                logger.info(f"TTS Stream: id={curr_task.id} req={req}")
                app.seg_manager.add_text(curr_task.id, req.text)
                if req.done:
                    app.seg_manager.add_text(curr_task.id, None)
                    await running_task
                    curr_task = None
                    running_task = None

        except BaseException as e:
            if curr_task is not None:
                if curr_task.id in app.tasks:
                    app.seg_manager.add_text(curr_task.id, None)
                curr_task = None
            if running_task is not None:
                if not running_task.done():
                    running_task.cancel()
                    await running_task
                running_task = None
            if isinstance(e, WebSocketDisconnect):
                break
            logger.error(f"Error in TTS (WebSocket):\n{traceback.format_exc()}")
            await websocket.send_json({"error": True, "message": whats_wrong_with(e)})
