from typing import TypedDict


class AddMetaDataParameter(TypedDict):
    user_id: int
    dataset_id: int
    prompt: str
    llm_response: str
    logic_action: str
    processing_history: str
    target_column_name: str


class UpdateMetaDataStatusParameter(TypedDict):
    user_id: int
    dataset_id: int
    status: str
    applier_status: str
