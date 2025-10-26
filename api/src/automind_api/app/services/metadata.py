from automind_api.app.models.dataset import ViewDataSource
from automind_api.app.models.view_sp import AvailableView, ViewMetaData
from automind_api.app.repositories.i3s import get_view_by_id


def verify_metadata(dataset_id: int) -> bool:
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

    if metadata_view.get("source_updated") is None:
        return True

    if metadata_view.get("source_updated") >= data_source_view.get(
        "updated_at"
    ):
        return False

    return True
