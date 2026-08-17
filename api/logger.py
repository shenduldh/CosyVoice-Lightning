import sys
from loguru import logger
from utils import path_to_root
from datetime import datetime
import os


def configure_logger():
    os.environ["COMPRESSED_TENSORS_LOG_DISABLED"] = "1"

    # remove default loggers
    logger.remove()

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # log to file
    log_dir = path_to_root("logs", f"{host}-{port}_{now_time}")
    os.environ["LOG_DIR"] = log_dir
    log_path = os.path.join(log_dir, "frontend.log")
    format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>{name}:{line}</cyan> - <level>{message}</level>"
    logger.add(
        log_path,
        format=format,
        level="INFO",
        rotation="10 MB",
        compression="zip",
        retention="1 months",
        encoding="utf-8",
        enqueue=True,
        watch=True,
    )

    # log to console
    logger.add(sys.stdout, format=format, level="INFO", colorize=True, enqueue=True)
