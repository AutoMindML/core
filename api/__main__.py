import asyncio

from automind.utils.console import rc, read_output

from automind_api.configs import get_config

config = get_config("api")

api_args = [
    "uv",
    "run",
    "-m",
    "uvicorn",
    "src.automind_api.main:app",
    # "--reload",
    "--host",
    config["host"],
    "--port",
    config["port"],
]

mindsdb_python = "../../mindsdb/.venv/Scripts/python.exe"
mindsdb_args = [
    mindsdb_python,
    "-m",
    "mindsdb",
    "--config",
    "./src/automind_api/configs/mindsdb.json",
    # "--no_studio",
]

api_name = "AutoMind Core"
mindsdb_name = "MindsDB"


async def run_api():
    process = await asyncio.create_subprocess_exec(
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
