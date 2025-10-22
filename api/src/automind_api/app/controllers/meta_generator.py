from automind.data_utils import MetaGenerator
from fastapi import APIRouter, Request, Response
from starlette import status

from automind_api.app.repositories.dataset import get_dataset

meta_generator_router = APIRouter()


@meta_generator_router.post("/{dataset_id}")
def generate_meta_generator_prompt(
    dataset_id: int, req: Request, res: Response
):
    dataset = get_dataset(
        dataset_id,
        req.state.user_id,
    )

    if dataset is None:
        res.status_code = status.HTTP_404_NOT_FOUND
        return {
            "state": 2,
            "message": "target dataset is not exists or user has no permission",
            "new_id": None,
        }

    meta_generator = MetaGenerator(dataset.get("table"), target_column="")
    meta_generator.extract_metadata()
    prompt = meta_generator.generate_llm_query()


@meta_generator_router.get("/{dataset_id}")
def get_meta_generator_prompt(dataset_id: int):
    pass
