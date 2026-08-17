import requests
import click


@click.command()
@click.option("--ip")
@click.option("--port")
@click.option("--cache_dir", default="./")
@click.option("--filename", default=None)
@click.option("--prompt_ids", default="")
def main(ip, port, cache_dir, filename, prompt_ids: str):
    prompt_ids = [] if prompt_ids == "" else [i.strip() for i in prompt_ids.split(",")]
    res = requests.post(f"http://{ip}:{port}/cache/save", json={"cache_dir": cache_dir, "filename": filename, "prompt_ids": prompt_ids})
    print(res.json())


if __name__ == "__main__":
    main()
