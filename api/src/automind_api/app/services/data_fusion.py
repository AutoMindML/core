from automind.data_utils.data_fusion_module import DataFusionModule

from automind_api.app.models.data_fusion import (
    CreateDataFusionDict,
)


def create_data_fusion(opts: CreateDataFusionDict):
    dfm = DataFusionModule()

    for dataset_id in opts["dataset_ids"]:
        pass
