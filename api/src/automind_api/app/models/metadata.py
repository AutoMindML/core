from typing import TypedDict


class AddMetaDataParameter(TypedDict):
    user_id: int
    dataset_id: int
    prompt: str
