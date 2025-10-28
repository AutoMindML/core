from typing import TypedDict


class ApplierGenerateNewDatasetParameter(TypedDict):
    name: str
    des: str
    user_id: int
    md5: str
    origin_dataset_id: int
