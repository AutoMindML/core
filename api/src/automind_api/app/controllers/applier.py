from fastapi import APIRouter, Request, Response
from starlette import status

from automind_api.app.services.file import generate_file_response
from automind_api.app.services.metadata import get_logic_applier_by_dataset_id

applier_router = APIRouter()


@applier_router.get("/{dataset_id}/preview")
def preview(dataset_id: int, req: Request, res: Response):
    applier = get_logic_applier_by_dataset_id(
        dataset_id, req.state.user_id, limit=20
    )

    if applier is None:
        res.status_code = status.HTTP_409_CONFLICT
        return {
            "state": 2,
            "message": "dataset is not found or target column not be specified or llm response not exists",
        }

    applier.apply_llm_recommendations()
    return generate_file_response(applier.get_processed_df(), res)


@applier_router.post("/apply")
def apply():
    pass


@applier_router.post("/save")
def save():
    pass


# @applier_router.get("/status")
# def status():
#     pass
