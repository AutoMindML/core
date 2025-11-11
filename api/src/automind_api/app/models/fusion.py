from typing import List, TypedDict

from pydantic import BaseModel


class InitDataFusionBody(BaseModel):
    name: str
    des: str


class SaveDataFusionBody(BaseModel):
    fusion_id: int
    target_dataset_id: int
    dataset_ids: List[str]
    primary_keys: List[str]
    relationships: List[str]
    position: List[object]


class SaveDataFusion(TypedDict):
    user_id: int
    fusion_id: int
    target_dataset_id: int
    dataset_ids: str
    primary_keys: str
    relationships: str
    position: str


class ViewDataFusion(TypedDict):
    dataset_ids: str
    relationships: str
    primary_keys: str
    fusion_id: int
    target_dataset_id: int
    position: str


class MergeDataFusion(TypedDict):
    fusion_id: int
    user_id: int
    md5: str
    rows: int
    cols: int
    col_names: str
    col_types: str
    size: float
    size_unit: str
    quality: float
