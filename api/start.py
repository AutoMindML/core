import asyncio

from automind.utils.console import rc, read_output
from dotenv import dotenv_values, load_dotenv

load_dotenv()

env = dotenv_values()


api_args = [
    "run",
    "-m",
    "uvicorn",
    "src.automind_api.main:app",
    # "--reload",
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
]

mindsdb_python = "../../mindsdb/.venv/Scripts/python.exe"
mindsdb_args = [
    "-m",
    "mindsdb",
    "--config",
    "./src/automind_api/configs/mindsdb.json",
    "--no_studio",
]

api_name = "AutoMind Core"
mindsdb_name = "MindsDB"


async def run_api():
    process = await asyncio.create_subprocess_exec(
        "uv",
        *api_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    io = asyncio.create_task(
        read_output(process.stdout, api_name, "bold red"), name=api_name
    )

    try:
        await process.wait()
    finally:
        io.cancel()
        process.terminate()


async def run_mindsdb():
    process = await asyncio.create_subprocess_exec(
        mindsdb_python,
        *mindsdb_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    io = asyncio.create_task(
        read_output(process.stdout, mindsdb_name, "bold cyan"),
        name=mindsdb_name,
    )

    try:
        await process.wait()
    finally:
        io.cancel()
        process.terminate()


async def main():
    api_process = run_api()
    mindsdb_process = run_mindsdb()

    try:
        await asyncio.gather(api_process, mindsdb_process)
    except asyncio.CancelledError:
        pass
    finally:
        rc.print("Terminating subprocesses...")

        api_process.close()
        mindsdb_process.close()

        rc.print("All subprocesses terminated.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rc.print("Received KeyboardInterrupt.")
