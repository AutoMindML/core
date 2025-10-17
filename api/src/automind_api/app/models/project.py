from pydantic import BaseModel


class ProjectAddRequest(BaseModel):
    name: str
    des: str


class ProjectDeleteRequest(BaseModel):
    cid: int
