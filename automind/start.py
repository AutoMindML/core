import asyncio

from dotenv import dotenv_values, load_dotenv

from automind.console import read_output
from automind.server import wait_until_server_ready

load_dotenv()

env = dotenv_values()

chainlit_host = env.get("CHAINLIT_HOST")
chainlit_port = env.get("CHAINLIT_PORT")
mcp_client_entry = env.get("MCP_CLIENT_ENTRY")

assert mcp_client_entry is not None
assert chainlit_host is not None
assert chainlit_port is not None

chainlit_url = f"http://{chainlit_host}:{chainlit_port}"

chainlit_args = [
    "run",
    "-m",
    "chainlit",
    "run",
    "-h",
    "-w",
    mcp_client_entry,
    "--host",
    chainlit_host,
    "--port",
    chainlit_port,
]


async def run_chainlit():
    process = await asyncio.create_subprocess_exec(
        "uv",
        *chainlit_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    if process.stdout is not None:
        asyncio.create_task(read_output(process.stdout, "Chainlit"))

    return process


async def main():
    process = await run_chainlit()
    await wait_until_server_ready(chainlit_url)

    await process.wait()


if __name__ == "__main__":
    asyncio.run(main())
