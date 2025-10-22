from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential

from automind_api.configs import get_config

# doc: https://azuresdkdocs.z19.web.core.windows.net/python/azure-ai-language-conversations/latest/azure.ai.language.conversations.html
config = get_config("private", "azure_ai")
endpoint = config["endpoint"]
credential = AzureKeyCredential(config["api_key"])
client = ConversationAnalysisClient(endpoint, credential)
# result = client.analyze_conversation()
