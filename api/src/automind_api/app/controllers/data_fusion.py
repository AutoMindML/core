from automind_api.app.models.data_fusion import DataFusionBody
from fastapi import APIRouter, Request

data_fusion_router = APIRouter()


@data_fusion_router.get("/preview")
def data_fusion_preview():
    pass


@data_fusion_router.post("/merge")
def data_fusion_merge(
    body: DataFusionBody,
    req: Request
):
    user_id = req.state.user_id


@data_fusion_router.post("/feature/generate")
def data_fusion_feature():
    pass


@data_fusion_router.post("/clear")
def data_fusion_clear():
    pass
