import json

from automind.data_utils import MetaGenerator
from fastapi import APIRouter, Request, Response

from automind_api.app.models.metadata import (
    AddMetaDataParameter,
    UpdateMetaDataStatusParameter,
)
from automind_api.app.models.view_sp import (
    AvailableSP,
)
from automind_api.app.repositories.i3s import exec_mutation_sp
from automind_api.app.services.azure import query_azure_openai
from automind_api.app.services.dataset import verify_dataset
from automind_api.app.services.metadata import (
    get_metadata_view,
    verify_metadata,
    verify_metadata_target_column,
)

metadata_router = APIRouter()


@metadata_router.post("/{dataset_id}")
def generate_metadata_prompt(
    dataset_id: int,
    req: Request,
    res: Response,
    target_column: str = "",
    force: bool = False,
):
    if (not verify_metadata(dataset_id)) and (not force):
        return {"state": 0, "message": "no change", "new_id": None}

    if target_column == "":
        target_column = verify_metadata_target_column(dataset_id)

    ret = verify_dataset(dataset_id, req.state.user_id, res, limit=-1)
    if ret.get("state") != 0:
        return ret

    update_metadata_status_opt: UpdateMetaDataStatusParameter = {
        "dataset_id": dataset_id,
        "user_id": req.state.user_id,
        "status": "generating",
        "applier_status": "unavailable",
    }

    exec_mutation_sp(
        AvailableSP.update_metadata_status, update_metadata_status_opt
    )

    meta_generator = MetaGenerator(
        ret.get("table"), target_column=target_column
    )
    meta_generator.extract_metadata()
    prompt = meta_generator.generate_llm_query()

    llm_response = query_azure_openai([prompt])

    add_or_update_metadata_opt: AddMetaDataParameter = {
        "dataset_id": dataset_id,
        "user_id": req.state.user_id,
        "prompt": prompt,
        "llm_response": json.dumps(llm_response.model_dump()),
        "target_column_name": target_column,
        "logic_action": "[]",
        "processing_history": "[]",
    }

    return exec_mutation_sp(
        AvailableSP.add_or_update_metadata, add_or_update_metadata_opt
    )


@metadata_router.get("/{dataset_id}")
def get_metadata_prompt(dataset_id: int, req: Request, res: Response):
    ret = verify_dataset(dataset_id, req.state.user_id, res)

    if ret.get("state") != 0:
        return ret

    metadata_view = get_metadata_view(dataset_id)

    if metadata_view.get("metadata_id") is None:
        return {"state": 2, "message": "metadata is not exists"}

    return {
        "state": 0,
        "message": "metadata is ready",
        "content": metadata_view.get("prompt"),
    }


@metadata_router.get("/{dataset_id}/status")
def get_metadata_status(dataset_id: int, req: Request, res: Response):
    ret = verify_dataset(dataset_id, req.state.user_id, res)

    if ret.get("state") != 0:
        return ret

    metadata_view = get_metadata_view(dataset_id)

    return {
        "status": 0,
        "message": "get status sucessfully",
        "content": {
            "metadata_status": metadata_view.get("status"),
            "applier_status": metadata_view.get("applier_status"),
        },
    }
