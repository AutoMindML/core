from typing import TypedDict

from automind.models.preprocessing import (
    DataCleaningOptions,
    FeatureEngineeringOptions,
)
from pydantic import BaseModel


class ApplierGenerateNewDatasetParameter(TypedDict):
    name: str
    des: str
    user_id: int
    md5: str
    origin_dataset_id: int


class ApplierProcessingBody(BaseModel):
    only_cleaning: bool
    data_cleaning_options: DataCleaningOptions
    feature_engineering_options: FeatureEngineeringOptions
