import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pandas import DataFrame

from automind_api.app.models.dataset import (
    dtype_map,
)
from automind_api.app.models.fusion import (
    InitDataFusionBody,
    MergeDataFusion,
    SaveDataFusion,
    SaveDataFusionBody,
    ViewDataFusion,
)
from automind_api.app.models.view_sp import AvailableView
from automind_api.app.repositories.dataset import dataset_to_mindsdb
from automind_api.app.repositories.i3s import exec_mutation_sp, get_view_by_id
from automind_api.app.services.file import (
    calculate_dataframe_md5,
    generate_file_response,
)
from automind_api.app.services.fusion import (
    create_data_fusion_module,
    verify_fusion_id,
)

fusion_router = APIRouter()


@fusion_router.post("/init")
def init_data_fusion(body: InitDataFusionBody, req: Request):
    opts = body.model_dump()
    opts["user_id"] = req.state.user_id
    return exec_mutation_sp("[dbo].[xp_init_dfm]", opts)


@fusion_router.put("/save")
def save_data_fusion(body: SaveDataFusionBody, req: Request):
    opts: SaveDataFusion = {
        "target_dataset_id": body.target_dataset_id,
        "fusion_id": body.fusion_id,
        "dataset_ids": str.join(",", body.dataset_ids),
        "primary_keys": str.join(",", body.primary_keys),
        "relationships": json.dumps(body.relationships),
        "position": json.dumps(body.position),
        "user_id": req.state.user_id,
    }

    return exec_mutation_sp("[dbo].[xp_update_dfm]", opts)


# fetch saved dataset and relationships by user selected
@fusion_router.get("/{fusion_id}")
def preview_data_fusion(fusion_id: Annotated[int, Depends(verify_fusion_id)]):
    view: ViewDataFusion = get_view_by_id(
        AvailableView.fusion,
        {"id": fusion_id, "id_col_name": "fusion_id"},
        ViewDataFusion,
    )
    return view


# fetch saved dataset and relationships by user selected and generate deep feature
@fusion_router.get("/{fusion_id}/generate")
def data_fusion_generate_feature(
    fusion_id: Annotated[int, Depends(verify_fusion_id)],
    req: Request,
    res: Response,
):
    dfm = create_data_fusion_module(fusion_id, req.state.user_id, 10)
    dfm.apply_dfs()
    deep_feature_df = dfm.get_deep_feature_dataframe()

    if deep_feature_df is not None:
        return generate_file_response(deep_feature_df, res)


# merge user selected dataset and relationships and generate deep feature
@fusion_router.post("/{fusion_id}/merge")
def merge_data_fusion(
    fusion_id: Annotated[int, Depends(verify_fusion_id)], req: Request
):
    dfm = create_data_fusion_module(fusion_id, req.state.user_id, -1)
    dfm.apply_dfs()
    deep_feature_df: DataFrame = dfm.get_deep_feature_dataframe()

    if deep_feature_df is not None:
        new_column_mapping = {
            col: col.replace("(", "_").replace(")", "").replace(".", "_")
            for col in deep_feature_df.columns
        }
        deep_feature_df.rename(columns=new_column_mapping, inplace=True)
        md5 = calculate_dataframe_md5(deep_feature_df).upper()
        rows, cols = deep_feature_df.shape
        new_column_mapping = {
            col: col.replace("(", "_").replace(")", "").replace(".", "_")
            for col in deep_feature_df.columns
        }
        deep_feature_df.rename(columns=new_column_mapping, inplace=True)
        col_names = deep_feature_df.columns.to_list()
        col_types = [
            dtype_map.get(str(dt), str(dt))
            for _, dt in deep_feature_df.dtypes.items()
        ]
        opts: MergeDataFusion = {
            "user_id": req.state.user_id,
            "fusion_id": fusion_id,
            "md5": md5,
            "rows": rows,
            "cols": cols,
            "col_names": ",".join(col_names),
            "col_types": ",".join(col_types),
            "size": float(
                deep_feature_df.memory_usage(index=False, deep=True).sum()
            ),
            "size_unit": "bytes",
            "quality": 0.0,
        }
        sp_message = exec_mutation_sp("[dbo].[xp_data_fusion_merge]", opts)
        dataset_to_mindsdb(deep_feature_df, md5)

        return sp_message
