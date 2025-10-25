from automind.data_utils import MetaGenerator
from fastapi import APIRouter, Request, Response

from automind_api.app.models.metadata import AddMetaDataParameter
from automind_api.app.models.view_sp import (
    AvailableSP,
    AvailableView,
    ViewMetaData,
)
from automind_api.app.repositories.i3s import exec_mutation_sp, get_view_by_id
from automind_api.app.services.dataset import verify_dataset
from automind_api.app.services.meta_generator import verify_metadata

meta_generator_router = APIRouter()


@meta_generator_router.post("/{dataset_id}")
def generate_meta_generator_prompt(
    dataset_id: int,
    target_column: str,
    req: Request,
    res: Response,
    force: bool = False,
):
    if (not verify_metadata(dataset_id)) and (not force):
        return {"state": 0, "message": "no change", "new_id": None}

    ret = verify_dataset(dataset_id, req.state.user_id, res, limit=-1)
    if ret.get("state") != 0:
        return ret

    meta_generator = MetaGenerator(
        ret.get("table"), target_column=target_column
    )
    meta_generator.extract_metadata()
    prompt = meta_generator.generate_llm_query()

    opt: AddMetaDataParameter = {
        "dataset_id": dataset_id,
        "user_id": req.state.user_id,
        "prompt": prompt,
    }

    return exec_mutation_sp(AvailableSP.add_or_update_metadata, opt)


@meta_generator_router.get("/{dataset_id}")
def get_meta_generator_prompt(dataset_id: int, req: Request, res: Response):
    ret = verify_dataset(dataset_id, req.state.user_id, res)

    if ret.get("state") != 0:
        return ret

    metadata_view: ViewMetaData = get_view_by_id(
        AvailableView.metadata,
        {"id": dataset_id, "id_col_name": "metadata_id"},
        ViewMetaData,
    )

    if metadata_view.get("metadata_id") is None:
        return {"state": 2, "message": "metadata is not exists"}

    return {
        "state": 0,
        "message": "metadata is ready",
        "content": metadata_view.get("prompt"),
    }
