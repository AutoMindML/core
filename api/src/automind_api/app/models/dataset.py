from typing import Dict, List, Literal, TypedDict

from pandas import DataFrame
from pydantic import BaseModel

DatasetType = Literal["file", "database"]


class DatasetReturn(TypedDict):
    table: DataFrame
    columns: List[str]


class AddDatabaseBody(BaseModel):
    name: str
    des: str
    engine: str
    connection_args: Dict[str, str | int]
