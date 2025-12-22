from typing import Optional, TypedDict

from automind.models.preprocessing import (
    DataCleaningOptions,
    FeatureEngineeringOptions,
    TaskOptions,
)
from pydantic import BaseModel


class ApplierGenerateNewDatasetParameter(TypedDict):
    name: str
    des: str
    user_id: int
    md5: str
    origin_dataset_id: int
    rows: int
    cols: int
    col_names: str
    col_types: str
    size: float
    size_unit: str
    quality: float


class ApplierProcessingBody(BaseModel):
    only_cleaning: Optional[bool] = False
    data_cleaning_options: Optional[DataCleaningOptions] = None
    feature_engineering_options: Optional[FeatureEngineeringOptions] = None
    task_options: Optional[TaskOptions] = None
