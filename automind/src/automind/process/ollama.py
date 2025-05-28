import asyncio
import enum
import queue
import subprocess
from typing import Optional

import ollama
from dotenv import dotenv_values, load_dotenv
from ollama import chat

from automind.console import ANSI_ESCAPE, SPINNER_SYMBOLS

load_dotenv()
env = dotenv_values()


class AvailibleModel(enum.Enum):
    deepseek_r1_14b = "deepseek-r1:14b"
    gemma3_12b = "gemma3:12b"
    mistral_7b = "mistral:7b"


CMD = "ollama"
END_OF_STREAM = "<<END_OF_STREAM>>"
DEFAULT_MODEL = AvailibleModel.mistral_7b.value
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
    buffer = ""

    if process.stdout is not None:
        while True:
            line = process.stdout.readline()

            print(line)

            cleaned_line = ANSI_ESCAPE.sub("", line.decode().strip())
            cleaned_line = SPINNER_SYMBOLS.sub("", cleaned_line)
            cleaned_line = cleaned_line.lstrip() + "\n"

            buffer += cleaned_line

            if env.get("OLLAMA_DEBUG") is not None:
                print(cleaned_line, flush=True)

            # detect empty line or other ending
            if buffer.endswith("\n\n\n"):
                break

        if env.get("OLLAMA_DEBUG") is not None:
            print(END_OF_STREAM, flush=True)

        buffer += END_OF_STREAM

    return buffer


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


def run_ollama_by_official_api(prompt: str, model: str = DEFAULT_MODEL):
    try:
        ollama.show(model)
    except ollama.ResponseError:
        print("model is not exists, try to pull model...")
        ollama.pull(model)
        print("model pull completed.")

    buffer = ""

    stream = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        if chunk.message.content:
            buffer += chunk.message.content

        if env.get("OLLAMA_DEBUG") is not None:
            print(chunk.message.content, end="", flush=True)

    return buffer


async def main():
    process = await test_ollama()

    await process.wait()


if __name__ == "__main__":
    asyncio.run(main())
