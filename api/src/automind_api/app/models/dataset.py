from datetime import date
from typing import Dict, List, Literal, TypedDict

from pandas import DataFrame
from pydantic import BaseModel

DatasetType = Literal["file", "database", "fusion"]


class DatasetReturn(TypedDict):
    table: DataFrame
    columns: List[str]


class DeleteDatasetParameter(TypedDict):
    user_id: int
    dataset_id: int


class AddDatabaseBody(BaseModel):
    name: str
    des: str
    engine: str
    connection_args: Dict[str, str | int]


class ViewDataSource(TypedDict):
    oid: int
    source_type: Literal["file", "database", "fusion"]
    name: str
    description: str
    md5: str
    created_at: date
    updated_at: date
    used_status: int
    is_hided: int
    is_deleted: int
    cid: int
    owner_mid: int
