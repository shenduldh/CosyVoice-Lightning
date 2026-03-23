import sys
from loguru import logger
from datetime import datetime
import os


os.environ["COMPRESSED_TENSORS_LOG_DISABLED"] = "1"


def configure_logger():
    # remove default loggers
    logger.remove()

    # log to file
    filename = f"{os.environ['HOST']}-{os.environ['PORT']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{name}:{line}</cyan> - <level>{message}</level>"
    logger.add(
        f"logs/{filename}.log",
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
    logger.add(
        sys.stdout,
        format=format,
        level="INFO",
        colorize=True,
        enqueue=True,
    )
