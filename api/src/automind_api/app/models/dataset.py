from datetime import date
from typing import Dict, List, Literal, Optional, TypedDict

from pandas import DataFrame
from pydantic import BaseModel

from automind_api.app.models.i3s import BasePostResponse

DatasetType = Literal["file", "database", "fusion"]

dtype_map = {
    "int64": "int",
    "float64": "float",
    "object": "string",
    "bool": "bool",
    "datetime64[ns]": "datetime",
    "string": "string",
}


class DatasetReturn(TypedDict):
    table: DataFrame
    columns: List[str]


class VerifyDatasetResponse(BasePostResponse):
    table: Optional[DataFrame]
    columns: Optional[List[str]]


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
    rows: int
    cols: int
    col_names: str
    col_types: str
    size: float
    size_unit: str
    quality: float


class AddDatasetFileParameter(TypedDict):
    user_id: int
    name: str
    des: str
    md5: str
    rows: int
    cols: int
    col_names: str
    col_types: str
    size: float
    size_unit: str
    quality: float
