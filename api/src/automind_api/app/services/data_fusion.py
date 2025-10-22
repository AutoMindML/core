import json

from automind.data_utils.data_fusion_module import DataFusionModule
from fastapi import HTTPException, status

from automind_api.app.models.data_fusion import (
    ViewDataFusion,
)
from automind_api.app.models.dataset import ViewDataSource
from automind_api.app.repositories.dataset import get_dataset
from automind_api.app.repositories.i3s import get_object_info, get_view_by_id


def create_data_fusion_module(
    fusion_id: int, user_id: int, data_size_limit: int = 20
):
    view: ViewDataFusion = get_view_by_id(
        "[dbo].[vd_data_fusion]",
        {"id": fusion_id, "id_col_name": "fusion_id"},
        ViewDataFusion,
    )

    target_dataset_id = view.get("target_dataset_id")

    # "d1,d2,d3"
    dataset_ids = view.get("dataset_ids").split(",")

    # "d1pk,d2pk,d3pk"
    dataset_pks = view.get("primary_keys").split(",")

    # ["d1,pk;d2,fk", "d3,pk;d4,fk"]
    relationships: list[str] = json.loads(view.get("relationships"))

    dataset_names = {
        dataset_id: get_object_info(int(dataset_id)).get("CName")
        + "_"
        + dataset_id
        for dataset_id in dataset_ids
    }

    dfm = DataFusionModule(
        str(view.get("fusion_id")), dataset_names.get(str(target_dataset_id))
    )

    for i in range(len(dataset_ids)):
        dataset_id = dataset_ids[i]
        pk = dataset_pks[i]
        dataset = get_dataset(
            int(dataset_id), user_id, "fusion", data_size_limit
        )

        if dataset is not None:
            dfm.add_entity(
                dataset.get("table"),
                dataset_names.get(dataset_id, ""),
                pk,
            )

    for relationship in relationships:
        entitys = relationship.split(";")
        e1 = entitys[0].split(",")
        e2 = entitys[1].split(",")
        e1_id = e1[0]
        e1_pk = e1[1]
        e2_id = e2[0]
        e2_pk = e2[1]

        dfm.add_relationship(
            dataset_names.get(e1_id, ""),
            e1_pk,
            dataset_names.get(e2_id, ""),
            e2_pk,
        )

    return dfm


def verify_fusion_id(fusion_id: int):
    view: ViewDataSource = get_view_by_id(
        "[dbo].[vd_Data_Source]",
        {"id": fusion_id, "id_col_name": "oid"},
        ViewDataSource,
    )

    if view.get("source_type") == "fusion":
        return fusion_id

    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        "this data is not for fusion task or data is not exists",
    )
