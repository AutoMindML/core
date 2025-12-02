import json

from automind.data_utils.logic_applier import LogicApplier
from fastapi import HTTPException
from openai.types.chat import ChatCompletion
from starlette import status

from automind_api.app.models.dataset import ViewDataSource
from automind_api.app.models.view_sp import AvailableView, ViewMetaData
from automind_api.app.repositories.dataset import get_dataset
from automind_api.app.repositories.i3s import get_view_by_id


def verify_metadata(dataset_id: int, target_column: str) -> bool:
    data_source_view: ViewDataSource = get_view_by_id(
        AvailableView.dataset,
        {"id": dataset_id, "id_col_name": "oid"},
        ViewDataSource,
    )

    metadata_view: ViewMetaData = get_view_by_id(
        AvailableView.metadata,
        {"id": dataset_id, "id_col_name": "metadata_id"},
        ViewMetaData,
    )

    if metadata_view.get("target_column_name") != target_column:
        return True

    if metadata_view.get("source_updated") is None:
        return True

    if metadata_view.get("source_updated") >= data_source_view.get(
        "updated_at"
    ):
        return False

    return True


def verify_metadata_target_column(dataset_id: int):
    metadata_view: ViewMetaData = get_view_by_id(
        AvailableView.metadata,
        {"id": dataset_id, "id_col_name": "metadata_id"},
        ViewMetaData,
    )

    if metadata_view.get("target_column_name", "") == "":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "target column is not specified"
        )

    return metadata_view.get("target_column_name")


def get_metadata_view(dataset_id: int):
    metadata_view: ViewMetaData = get_view_by_id(
        AvailableView.metadata,
        {"id": dataset_id, "id_col_name": "metadata_id"},
        ViewMetaData,
    )

    return metadata_view


def get_logic_applier_by_dataset_id(
    dataset_id: int, user_id: int, limit: int = 20
):
    dataset = get_dataset(dataset_id, user_id, limit=limit)
    if dataset is None:
        return None

    metadata_view = get_metadata_view(dataset_id)

    if (metadata_view.get("target_column_name", "") == "") or (
        metadata_view.get("llm_response", "") == ""
    ):
        return None

    llm_response = json.loads(metadata_view.get("llm_response"))
    completion = ChatCompletion.model_validate(llm_response)
    content = completion.choices[0].message.content or ""

    applier = LogicApplier(
        dataset.get("table"), metadata_view.get("target_column_name"), content
    )

    logic_action = json.loads(metadata_view.get("logic_action", "[]"))
    applier.logic_actions = logic_action
    applier.parse_llm_response()

    return applier
