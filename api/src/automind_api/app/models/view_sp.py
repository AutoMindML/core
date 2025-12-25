from datetime import date
from typing import Literal, TypedDict

from automind_api.app.models.shared import Status


class AvailableView:
    dataset = "[dbo].[vd_Data_Source]"
    fusion = "[dbo].[vd_data_fusion]"
    metadata = "[dbo].[vd_metadata]"
    app_prediction = "[dbo].[vd_App_Prediction]"
    model = "[dbo].[vd_Model]"


class AvailableSP:
    add_or_update_metadata = "[dbo].[xp_add_metadata]"
    update_metadata_status = "[dbo].[xp_update_metadata_status]"
    applier_generate_new_dataset = "[dbo].[xp_applier_generate_new_dataset]"


class ViewMetaData(TypedDict):
    metadata_id: int
    prompt: str
    source_updated: date
    llm_response: str
    target_column_name: str
    logic_action: str
    processing_history: str
    status: Status
    applier_status: Literal["unavailable", "generating", "complete"]


class ViewModel(TypedDict):
    project_id: int
    model_id: int
    app_id: int
    api_key: str
    name: str
    description: str
    output_features: str
    input_features: str
    select_data_query: str
    created_at: date
    updated_at: date
    owner_mid: int
    active: int
    version: int
    status: str
    score: str
    predict: str
    learning_type: str
    task_type: str
    error: str
    training_time: float
    update_status: str
    training_options: str
    tag: str
    current_training_phase: int
    total_training_phases: int
    data_source_id: int
    data_source_md5: str
    data_source_type: str
    engine_id: int
    engine_md5: str
