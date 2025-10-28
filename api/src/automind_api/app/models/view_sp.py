from datetime import date
from typing import Literal, TypedDict


class AvailableView:
    dataset = "[dbo].[vd_Data_Source]"
    metadata = "[dbo].[vd_metadata]"


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
    status: Literal["unavailable", "generating", "complete"]
    applier_status: Literal["unavailable", "generating", "complete"]
