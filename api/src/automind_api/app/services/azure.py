from openai import AzureOpenAI

from automind_api.configs import get_config

config = get_config(
    "it108",
    "private",
)
endpoint = config["endpoint"]
key = config["api_key"]


# api version spec:
# https://github.com/Azure/azure-rest-api-specs/tree/main/specification/cognitiveservices/resource-manager/Microsoft.CognitiveServices
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version="2024-07-01-preview",
    azure_deployment="IT108_gpt35",
)

if __name__ == "__main__":
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "hello"}],  # pyright: ignore[reportArgumentType]
    )

    print(completion.choices[0].model_dump())
