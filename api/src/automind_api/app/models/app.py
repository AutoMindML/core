from pydantic import BaseModel, ConfigDict


class AddAppBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    project_id: int
    model_id: int
    name: str
    des: str


class DeleteAppBody(BaseModel):
    app_id: int
