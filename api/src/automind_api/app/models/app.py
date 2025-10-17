from pydantic import BaseModel, ConfigDict


class AddAppBody(BaseModel):
    project_id: int
    model_id: int
    name: str
    des: str
    model_config = ConfigDict(protected_namespaces=())


class DeleteAppBody(BaseModel):
    app_id: int
