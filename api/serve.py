import uvicorn
import os
import click
from dotenv import load_dotenv
import sys
import asyncio
import uvloop
from utils import path_to_root
from logger import configure_logger


@click.command()
@click.option("--env", default="./.env")
def main(env):
    sys.path.insert(0, path_to_root())
    sys.path.insert(0, path_to_root("tts_fast"))
    load_dotenv(env, override=True, encoding="utf-8")
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    configure_logger()
    uvicorn.run(
        "app:app",
        loop="none",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
