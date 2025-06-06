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

mindsdb_python = "../../mindsdb/.venv/Scripts/python.exe"
mindsdb_args = ["-m", "mindsdb", "--config", "config.json", "--no_studio"]


async def run_api():
    process = await asyncio.create_subprocess_exec(
        "uv",
        *api_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    if process.stdout is not None:
        asyncio.create_task(read_output(process.stdout, "API"))

    await process.wait()


async def run_mindsdb():
    process = await asyncio.create_subprocess_exec(
        mindsdb_python,
        *mindsdb_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    if process.stdout is not None:
        asyncio.create_task(read_output(process.stdout, "MINDSDB", "bold cyan"))

    await process.wait()


async def main():
    api_process = run_api()
    mindsdb_process = run_mindsdb()

    await asyncio.gather(api_process, mindsdb_process)


if __name__ == "__main__":
    asyncio.run(main())
