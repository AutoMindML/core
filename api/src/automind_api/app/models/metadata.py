from typing import TypedDict


class AddMetaDataParameter(TypedDict):
    user_id: int
    dataset_id: int
    prompt: str
    llm_response: str
    parsed_action: str
    parsed_history: str
    target_column_name: str
