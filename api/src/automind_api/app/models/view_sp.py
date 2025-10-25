from datetime import date
from typing import TypedDict


class AvailableView:
    dataset = "[dbo].[vd_Data_Source]"
    metadata = "[dbo].[vd_metadata]"


class AvailableSP:
    add_or_update_metadata = "[dbo].[xp_add_metadata]"


class ViewMetaData(TypedDict):
    metadata_id: int
    prompt: str
    source_updated: date
