from typing import List, TypedDict

from pydantic import BaseModel


class InitDataFusion(BaseModel):
    name: str
    des: str


class InitDataFusionDict(TypedDict):
    user_id: int
    name: str
    des: str


class CreateDataFusion(BaseModel):
    target_id: int
    dataset_ids: List[int]
    relationships: List[List[str]]


class CreateDataFusionDict(TypedDict):
    target_id: int
    dataset_ids: List[int]
    relationships: List[List[str]]
