from typing import Optional, TypedDict


class ObjectInfo(TypedDict):
    CName: str


class BasePostResponse(TypedDict):
    state: int
    message: str


class ExecMutationSpOutput(BasePostResponse):
    new_id: Optional[int]


class GetViewCommonOpts(TypedDict):
    limit: int


class GetViewByIdOpts(TypedDict):
    id: int
    id_col_name: str
