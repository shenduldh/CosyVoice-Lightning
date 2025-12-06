import sys
from loguru import logger
from datetime import datetime
import os

# remove default loggers
logger.remove(None)

# log to file
prefix = f"{os.environ['HOST']}_{os.environ['PORT']}"
time = datetime.now().strftime("%Y-%m-%d")
filename = f"{prefix}_{time}"
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>{name}:{line}</cyan> - <level>{message}</level>"
logger.add(
    f"logs/{filename}.log",
    format=logger_format,
    level="INFO",
    rotation="1 day",
    compression="zip",
    retention="3 months",
    encoding="utf-8",
    enqueue=True,
)

# log to console
logger.add(
    sys.stdout,
    format=logger_format,
    level="INFO",
    enqueue=True,
)
