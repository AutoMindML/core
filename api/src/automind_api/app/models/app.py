from datetime import date
from typing import TypedDict

from pydantic import BaseModel, ConfigDict


class AddAppBody(BaseModel):
    project_id: int
    model_id: int
    name: str
    des: str
    model_config = ConfigDict(protected_namespaces=())


class DeleteAppBody(BaseModel):
    app_id: int


class ViewAppPrediction(TypedDict):
    project_id: int
    app_id: int
    app_type: str
    name: str
    description: str
    created_at: date
    updated_at: date
    owner_mid: int
    active_models: int
    api_key: str
    app_status: str
    deployment_id: str
