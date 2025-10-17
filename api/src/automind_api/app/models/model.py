from typing import Dict

from pydantic import BaseModel, ConfigDict


class ModelAddRequest(BaseModel):
    cid: int
    data_oid: int
    engine_oid: int
    name: str
    des: str
    predict: str
    tag: str
    select_data_query: str
    training_options: str


class ModelDeleteRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    project_id: int
    model_id: int


InputFeatures = list[Dict[str, float] | Dict[str, str]]


class ModelPredictionBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: int
    input_features: InputFeatures
