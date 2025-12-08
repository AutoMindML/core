from typing import Iterable, List

from openai import AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from automind.utils.config import get_config

config = get_config(
    "azure",
    "private",
)
endpoint = config.get("endpoint", "")
deployment = config.get("deployment", "")
api_key = config.get("api_key", "")
api_version = config.get("api_version", "")


# setup steps: https://studyhost.blogspot.com/2024/01/azure-openai.html
# api spec: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    azure_deployment=deployment,
)


def query_azure_openai(contents: List[str]) -> ChatCompletion:
    messages: Iterable[ChatCompletionMessageParam] = []

    for content in contents:
        messages.append({"role": "system", "content": content})

    completion = client.chat.completions.create(
        model=deployment, messages=messages
    )

    return completion


if __name__ == "__main__":
    completion = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": "hello"}],  # pyright: ignore[reportArgumentType]
    )

    print(completion.choices[0].model_dump())
