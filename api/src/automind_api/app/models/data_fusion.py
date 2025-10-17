from typing import List

from pydantic import BaseModel, ConfigDict


class DataFusionBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_ids = List[str]
    pass
