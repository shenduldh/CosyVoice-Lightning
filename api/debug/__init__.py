import os
import zipfile
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse
from pathlib import Path
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from typing import List
from utils import path_to_root
import hashlib
import shutil
from collections import defaultdict
import numpy as np
from datetime import datetime
from utils import save_audio
import json
import uuid
import re
import schedule
import multiprocessing
from pytz import timezone
import time


def debug_print(msg: str):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | debug - {msg}")


def clean_expired(root: Path, max_hours=24 * 14, date=None, this_dir: Path = None):
    if date is None:
        date = datetime.now()

    # root
    if this_dir is None:
        for subdir in root.iterdir():
            clean_expired(root, max_hours, date, subdir)
        debug_print("Clean expired done.")
        return False

    # tail
    if re.match(r"[0-9]{2}-[0-9]{2}_[0-9a-z]+", this_dir.name):
        year_month, day, hour, minute_second = this_dir.relative_to(root).parts
        year, month = year_month.split("-")
        minute, second = minute_second.split("_")[0].split("-")
        this_date = datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second)
        )
        hours_diff = (date - this_date).total_seconds() // 3600
        if hours_diff > max_hours:
            shutil.rmtree(this_dir, ignore_errors=True)
            return True
        return False

    # other
    deleted = 0
    total = 0
    for subdir in this_dir.iterdir():
        total += 1
        if clean_expired(root, max_hours, date, subdir):
            deleted += 1
    if deleted == total:
        shutil.rmtree(this_dir, ignore_errors=True)
        return True
    return False


def clean_temp(path: Path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    debug_print("Clean temp done.")


def clean_job(stop_event, saved_path, temp_path):
    debug_print("Clean task start.")
    tz = timezone("Asia/Shanghai")
    schedule.every().day.at("00:00", tz).do(clean_expired, saved_path)
    schedule.every().saturday.at("00:00", tz).do(clean_temp, temp_path)
    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)


def set_debug_routing(router: APIRouter):
    @router.get("/")
    async def index(request: Request):
        return debug.templates.TemplateResponse("index.html", {"request": request})

    @router.get("/files")
    async def files():
        return debug.get_file_tree()

    @router.get("/dl/file/{file_name:path}")
    async def download_file(file_name: str):
        file_path = debug.saved_path / file_name
        return FileResponse(file_path, filename=file_path.name)

    @router.get("/dl/folder/{folder_name:path}")
    async def download_folder(folder_name: str):
        md5 = hashlib.md5()
        md5.update(str(folder_name).encode())
        zip_path = str(debug.temp_path / md5.hexdigest()) + ".zip"
        folder_path = debug.saved_path / folder_name

        if not os.path.exists(zip_path):
            zip_folder(str(folder_path), zip_path)

        return FileResponse(
            zip_path, filename=f"{folder_path.name}.zip", media_type="application/zip"
        )

    @router.get("/del/{target_path:path}")
    async def delete(target_path: str):
        shutil.rmtree(debug.saved_path / target_path, ignore_errors=True)
        if not os.path.exists(debug.saved_path):
            os.makedirs(debug.saved_path, exist_ok=True)
        return {"status": "ok"}


class Debug:
    def __init__(self):
        self.templates = Jinja2Templates(path_to_root("api", "debug", "templates"))
        host_id = f"{os.environ['HOST']}_{os.environ['PORT']}"
        self.saved_path = Path(path_to_root("api", "debug", "saved", host_id))
        self.temp_path = Path(path_to_root("api", "debug", "temp", host_id))
        self.static_path = path_to_root("api", "debug", "static")
        self.data = defaultdict(lambda: {"chunks": [], "text": []})
        self.router = None
        self.enabled = False

    def add_chunk(self, req_id: str, audio_chunk: np.ndarray):
        if self.enabled:
            self.data[req_id]["chunks"].append(audio_chunk)

    def add_text(self, req_id: str, text_clips: List[str]):
        if self.enabled:
            self.data[req_id]["text"] += text_clips

    def save(self, req_id, tts_info, sample_rate):
        if self.enabled:
            year_month = tts_info.date.strftime("%Y-%m")
            day = tts_info.date.strftime("%d")
            hour = tts_info.date.strftime("%H")
            time = tts_info.date.strftime("%M-%S")
            saved_dir = os.path.join(
                self.saved_path, year_month, day, hour, f"{time}_{req_id}"
            )
            os.makedirs(saved_dir, exist_ok=True)

            if len(self.data[req_id]["chunks"]) > 0:
                full_audio = np.concatenate(self.data[req_id]["chunks"])
                saved_path = os.path.join(saved_dir, "audio.wav")
                save_audio(full_audio, saved_path, sample_rate)
            if len(self.data[req_id]["text"]) > 0:
                info = {
                    "speaker": tts_info.request.prompt_id,
                    "instruct_text": tts_info.request.instruct_text,
                    "resample_rate": tts_info.request.sample_rate,
                    "audio_format": tts_info.request.audio_format,
                    "response_seconds": (
                        datetime.now() - tts_info.date
                    ).total_seconds(),
                    "tts_text": self.data[req_id]["text"],
                }
                saved_path = os.path.join(saved_dir, "info.json")
                with open(saved_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=4)

    def set_enabled(self, is_enabled):
        self.enabled = is_enabled is True

    def get_file_tree(self, dir_path: Path = None, level=0):
        this_dir = self.saved_path if dir_path is None else dir_path
        tree = {
            "name": this_dir.name,
            "id": uuid.uuid4().hex,
            "type": "folder",
            "level": level,
            "children": [],
            "dl_href": f"{self.router.prefix}/dl/folder/{this_dir.relative_to(self.saved_path)}",
            "del_href": f"{self.router.prefix}/del/{this_dir.relative_to(self.saved_path)}",
        }
        for child in this_dir.iterdir():
            if child.is_dir():
                tree["children"].append(self.get_file_tree(child, level + 1))
            else:
                tree["children"].append(
                    {
                        "name": child.name,
                        "id": uuid.uuid4().hex,
                        "type": "file",
                        "level": level + 1,
                        "dl_href": f"{self.router.prefix}/dl/file/{child.relative_to(self.saved_path)}",
                    }
                )
        tree["children"].sort(key=lambda i: i["name"])
        return tree

    def on_startup(self):
        if self.enabled:
            os.makedirs(self.temp_path, exist_ok=True)
            os.makedirs(self.saved_path, exist_ok=True)

            self.clean_event = multiprocessing.Event()
            self.clean_process = multiprocessing.Process(
                target=clean_job,
                args=(self.clean_event, self.saved_path, self.temp_path),
                daemon=True,
            )
            self.clean_process.start()

    def on_destroy(self):
        if self.enabled:
            self.clean_event.set()
            self.clean_process.join(timeout=10)
            if self.clean_process.is_alive():
                self.clean_process.terminate()

            shutil.rmtree(self.temp_path, ignore_errors=True)
            shutil.rmtree(self.saved_path, ignore_errors=True)

    def patch(
        self, app: FastAPI, router_prefix="/debug", static_routing_name="debug_static"
    ):
        if self.enabled:
            self.router = APIRouter(prefix=router_prefix)
            set_debug_routing(self.router)
            app.mount(
                f"/{static_routing_name}",
                StaticFiles(directory=self.static_path),
                name=static_routing_name,
            )
            app.include_router(self.router)


debug = Debug()
