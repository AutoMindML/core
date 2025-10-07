from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

preprocessing_router = APIRouter()


class DataPreprocessingModel(BaseModel):
    data: List[dict]


@preprocessing_router.post("/cleaning")
def cleaning(req: DataPreprocessingModel):
    pass


@preprocessing_router.post("/engineering")
def engineering(req: DataPreprocessingModel):
    pass


@preprocessing_router.post("/outlier")
def outlier(req: DataPreprocessingModel):
    pass
