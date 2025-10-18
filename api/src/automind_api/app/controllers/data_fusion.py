from fastapi import APIRouter, Request

from automind_api.app.models.data_fusion import CreateDataFusion, InitDataFusion
from automind_api.app.repositories.i3s import exec_mutation_sp

data_fusion_router = APIRouter()


@data_fusion_router.post("/init")
def init_data_fusion(body: InitDataFusion, req: Request):
    opts = body.model_dump()
    opts["user_id"] = req.state.user_id
    return exec_mutation_sp("[dbo].[xp_init_dfm]", opts)


@data_fusion_router.delete("/{fusion_id}")
def delete_data_fusion():
    pass


# fetch saved dataset and relationships by user selected
@data_fusion_router.get("/preview")
def preview_data_fusion():
    pass


# save user selected dataset and relationships temporary
@data_fusion_router.post("/merge")
def data_fusion_merge(body: CreateDataFusion, req: Request):
    pass


@data_fusion_router.post("/feature/generate")
def data_fusion_generate_feature():
    pass
