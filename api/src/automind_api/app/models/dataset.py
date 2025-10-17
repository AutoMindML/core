from typing import Dict

from pydantic import BaseModel


class AddDatabaseModel(BaseModel):
    name: str
    des: str
    engine: str
    connection_args: Dict[str, str | int]
