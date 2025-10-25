from openai import AzureOpenAI

from automind_api.configs import get_config

config = get_config(
    "azure_ai",
    "private",
)
endpoint = config["endpoint"]
key = config["api_key"]


client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version="2024-07-01-preview",
    azure_deployment="gpt35_azure",
)

if __name__ == "__main__":
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "data scientist", "content": "hello"}],  # pyright: ignore[reportArgumentType]
    )

    print(completion.choices[0].message.content)
