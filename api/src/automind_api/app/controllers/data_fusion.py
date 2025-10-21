import json

from fastapi import APIRouter, Request, Response
from pandas import DataFrame

from automind_api.app.models.data_fusion import (
    InitDataFusionBody,
    MergeDataFusion,
    SaveDataFusion,
    SaveDataFusionBody,
)
from automind_api.app.repositories.dataset import dataset_to_mindsdb
from automind_api.app.repositories.i3s import exec_mutation_sp
from automind_api.app.services.data_fusion import create_data_fusion_module
from automind_api.app.services.file import (
    calculate_dataframe_md5,
    generate_file_response,
)

data_fusion_router = APIRouter()


@data_fusion_router.post("/init")
def init_data_fusion(body: InitDataFusionBody, req: Request):
    opts = body.model_dump()
    opts["user_id"] = req.state.user_id
    return exec_mutation_sp("[dbo].[xp_init_dfm]", opts)


@data_fusion_router.put("/save")
def save_data_fusion(body: SaveDataFusionBody, req: Request):
    opts: SaveDataFusion = {
        "target_dataset_id": body.target_dataset_id,
        "fusion_id": body.fusion_id,
        "dataset_ids": str.join(",", body.dataset_ids),
        "primary_keys": str.join(",", body.primary_keys),
        "relationships": json.dumps(body.relationships),
        "user_id": req.state.user_id,
    }

    return exec_mutation_sp("[dbo].[xp_update_dfm]", opts)


# fetch saved dataset and relationships by user selected
@data_fusion_router.get("/{fusion_id}/preview")
def preview_data_fusion(fusion_id: int, req: Request):
    dfm = create_data_fusion_module(fusion_id, req.state.user_id, 10)
    return dfm.get_entity_set_relationships()


# fetch saved dataset and relationships by user selected and generate deep feature
@data_fusion_router.post("/{fusion_id}/feature/generate")
def data_fusion_generate_feature(fusion_id: int, req: Request, res: Response):
    dfm = create_data_fusion_module(fusion_id, req.state.user_id, 10)
    dfm.apply_dfs()
    deep_feature_df = dfm.get_deep_feature_dataframe()

    if deep_feature_df is not None:
        return generate_file_response(deep_feature_df, res)


# merge user selected dataset and relationships and generate deep feature
@data_fusion_router.post("/{fusion_id}/merge")
def merge_data_fusion(fusion_id: int, req: Request, res: Response):
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
        opts: MergeDataFusion = {
            "user_id": req.state.user_id,
            "fusion_id": fusion_id,
            "md5": md5,
        }
        exec_mutation_sp("[dbo].[xp_data_fusion_merge]", opts)
        dataset_to_mindsdb(deep_feature_df, md5)
        return {"state": 0, "message": md5}
