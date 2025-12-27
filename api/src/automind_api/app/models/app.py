from datetime import date
from typing import TypedDict

from pydantic import BaseModel


class CreateAppBody(BaseModel):
    project_id: int
    name: str
    des: str


class CreateAppParameter(TypedDict):
    user_id: int
    project_id: int
    name: str
    des: str


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
