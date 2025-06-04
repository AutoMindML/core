import re
from asyncio import StreamReader

from rich.console import Console

console = Console()

ANSI_ESCAPE = re.compile(r"(?:\x1B[@-Z\\-_]|\x1B\[[0-?]*[ -/]*[@-~])")
SPINNER_SYMBOLS = re.compile(r"[\u2800-\u28FF]")


def print_format_output(name, content):
    console.print(f"[bold red][{name}][/bold red]: {content}")
    console.file.flush()


async def read_output(stream: StreamReader, name):
    while True:
        line = await stream.readline()

        if not line:
            break

        decoded_line = None

        try:
            decoded_line = line.decode("utf-8").strip()
        except UnicodeDecodeError:
            decoded_line = line.decode("big5").strip()

        print_format_output(name, decoded_line)
