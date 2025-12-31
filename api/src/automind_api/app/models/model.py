from typing import Dict, List

from pydantic import BaseModel, ConfigDict


class TrainModelBody(BaseModel):
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


InputFeatures = List[Dict[str, float | str | int]]


class ModelPredictionServiceBody(BaseModel):
    input: InputFeatures
    model_id: str
    limit: int = -1


class ModelPredictionBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    project_id: int
    model_id: int
    dataset_id: int
    limit: int = 20
