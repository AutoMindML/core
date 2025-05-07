from asyncio import StreamReader

from rich.console import Console

console = Console()


def print_format_output(name, content):
    console.print(f"[bold red][{name}][/bold red]: {content}")
    console.file.flush()


async def read_output(stream: StreamReader, name):
    while True:
        line = await stream.readline()
        if not line:
            break
        print_format_output(name, line.strip().decode("utf-8"))
