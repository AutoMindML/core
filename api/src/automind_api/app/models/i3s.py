from typing import TypedDict


class ObjectInfo(TypedDict):
    CName: str


class ExecMutationSpOutput(TypedDict):
    state: int
    message: str
