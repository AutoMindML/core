import asyncio
import enum
import queue
import subprocess
from typing import Optional

from automind.console import ANSI_ESCAPE, SPINNER_SYMBOLS


class AvailibleModel(enum.Enum):
    llama3_2 = "llama3.2"


CMD = "ollama"
END_OF_STREAM = "<<END_OF_STREAM>>"
DEFAULT_MODEL = AvailibleModel.llama3_2.value
TEST_PROMPT = "generate random python code."


async def test_ollama():
    args = ["run", DEFAULT_MODEL, TEST_PROMPT]

    process = await asyncio.create_subprocess_exec(
        CMD,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    if process.stdout is not None:
        async for line in process.stdout:
            decoded_line = line.decode().strip()
            print(decoded_line)

    return process


def run_ollama(prompt: str, model: str = DEFAULT_MODEL):
    process = subprocess.Popen(
        [CMD, "run", model, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    buffer_list = []

    if process.stdout is not None:
        buffer = ""

        while True:
            line = process.stdout.readline()

            cleaned_line = ANSI_ESCAPE.sub("", line.decode().strip())
            cleaned_line = SPINNER_SYMBOLS.sub("", cleaned_line)
            cleaned_line = cleaned_line.lstrip() + "\n"

            buffer += cleaned_line
            buffer_list.append(cleaned_line)

            # detect empty line or other ending
            if buffer.endswith("\n\n\n"):
                break

        buffer_list.append(END_OF_STREAM)

    return buffer_list


def ollama_stream(
    prompt: str, model: str = DEFAULT_MODEL, buffer_queue: Optional[queue.Queue] = None
):
    process = subprocess.Popen(
        [CMD, "run", model, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if process.stdout is not None:
        buffer = ""

        while True:
            line = process.stdout.readline()

            cleaned_line = ANSI_ESCAPE.sub("", line.decode().strip())
            cleaned_line = SPINNER_SYMBOLS.sub("", cleaned_line)
            cleaned_line = cleaned_line.lstrip() + "\n"

            if buffer_queue is not None:
                buffer_queue.put(cleaned_line)

            buffer += cleaned_line

            yield cleaned_line

            # detect empty line or other ending
            if buffer.endswith("\n\n\n"):
                break

        yield END_OF_STREAM

        if buffer_queue is not None:
            buffer_queue.put(END_OF_STREAM)


def buffer_stream(buffer_queue: queue.Queue):
    while True:
        line = buffer_queue.get()

        if line == END_OF_STREAM:
            break

        yield line


async def main():
    process = await test_ollama()

    await process.wait()


if __name__ == "__main__":
    asyncio.run(main())
