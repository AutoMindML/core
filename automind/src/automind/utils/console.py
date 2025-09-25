import re
from asyncio import StreamReader

from rich.console import Console

rc = Console()

ANSI_ESCAPE = re.compile(r"(?:\x1B[@-Z\\-_]|\x1B\[[0-?]*[ -/]*[@-~])")
SPINNER_SYMBOLS = re.compile(r"[\u2800-\u28FF]")


def print_format_output(name, content, color="bold red"):
    content = ANSI_ESCAPE.sub("", content)
    rc.print(f"[{color}][{name}][/{color}] -> {content}")
    rc.file.flush()


def apply_color(content: str, color: str = "bold red"):
    return f"[{color}]{content}[/{color}]"


async def read_output(stream: StreamReader, name, color=None):
    while not stream.at_eof():
        line = await stream.readline()

        if not line:
            break

        decoded_line = None

        try:
            decoded_line = line.decode("utf-8").strip()
        except UnicodeDecodeError:
            decoded_line = line.decode("big5").strip()

        if color:
            print_format_output(name, decoded_line, color)
        else:
            print_format_output(name, decoded_line)
