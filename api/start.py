import asyncio

from automind.console import read_output
from dotenv import dotenv_values, load_dotenv

load_dotenv()

env = dotenv_values()


api_args = [
    "run",
    "-m",
    "uvicorn",
    "src.api.main:app",
    # "--reload",
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
]


async def run_api():
    process = await asyncio.create_subprocess_exec(
        "uv",
        *api_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    if process.stdout is not None:
        asyncio.create_task(read_output(process.stdout, "API"))

    return process


async def main():
    process = await run_api()

    await process.wait()


if __name__ == "__main__":
    asyncio.run(main())
