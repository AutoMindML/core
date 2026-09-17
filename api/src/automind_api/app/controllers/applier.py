from automind.models.preprocessing import TaskType
from automind.pipeline import PreprocessingPipeline
from fastapi import APIRouter, Request, Response
from sklearn.model_selection import train_test_split

from automind_api.app.models.applier import (
    ApplierProcessingBody,
)
from automind_api.app.models.metadata import UpdateMetaDataStatusParameter
from automind_api.app.models.view_sp import AvailableSP
from automind_api.app.repositories.i3s import exec_mutation_sp, get_object_info
from automind_api.app.services.applier import (
    generate_409_conflict_response,
    generate_new_dataset_from_df,
)
from automind_api.app.services.file import (
    generate_file_response,
)
from automind_api.app.services.metadata import (
    get_logic_applier_by_dataset_id,
    get_metadata_view,
)

applier_router = APIRouter()


@applier_router.get("/{dataset_id}/preview/actions")
def applier_preview_actions(dataset_id: int, req: Request, res: Response):
    applier = get_logic_applier_by_dataset_id(
        dataset_id, req.state.user_id, limit=20
    )

    if applier is None:
        return generate_409_conflict_response(res)

    return {
        "state": 0,
        "message": "get actions successfully",
        "content": applier.logic_actions,
    }


@applier_router.post("/{dataset_id}/preview/processing")
def applier_preview_processing_result(
    dataset_id: int, req: Request, res: Response, body: ApplierProcessingBody
):
    applier = get_logic_applier_by_dataset_id(
        dataset_id, req.state.user_id, limit=20
    )

    if applier is None:
        return generate_409_conflict_response(res)

    applier.apply_llm_recommendations(
        body.data_cleaning_options,
        body.feature_engineering_options,
        body.task_options,
        only_cleaning=body.only_cleaning or False,
    )

    return generate_file_response(applier.get_processed_df(), res)


@applier_router.post("/{dataset_id}/save")
def applier_save_processing_result(
    dataset_id: int, req: Request, res: Response, body: ApplierProcessingBody
):
    applier = get_logic_applier_by_dataset_id(
        dataset_id, req.state.user_id, limit=-1
    )

    if applier is None:
        return generate_409_conflict_response(res)

    metadata_view = get_metadata_view(dataset_id)
    object_info = get_object_info(dataset_id)

    update_metadata_status_opt: UpdateMetaDataStatusParameter = {
        "dataset_id": dataset_id,
        "user_id": req.state.user_id,
        "status": metadata_view.get("status"),
        "applier_status": "generating",
    }

    exec_mutation_sp(
        AvailableSP.update_metadata_status, update_metadata_status_opt
    )

    try:
        source = applier.get_origin_df()
        target_column = applier.target_column
        if target_column is None or target_column not in source.columns:
            raise ValueError("target column is unavailable")

        task_type = applier.logic_actions[0].modeling_approaches[0].task_type
        stratify = (
            source[target_column] if task_type != TaskType.REGRESSION else None
        )
        train_source, val_source = train_test_split(
            source,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        fitted = PreprocessingPipeline(strict=True).fit(
            train_source,
            target_column,
            applier.llm_response,
            body.data_cleaning_options,
            body.feature_engineering_options,
            body.task_options,
        )
        prepared_train = fitted.fit_resample_training(train_source)
        prepared_val = fitted.transform(val_source)
        train_df = prepared_train.X.assign(
            **{target_column: prepared_train.y.values}
        )
        val_df = prepared_val.X.assign(
            **{target_column: prepared_val.y.values}
        )

        generate_new_dataset_from_df(
            train_df,
            dataset_id,
            req.state.user_id,
            object_info,
            "train",
            metadata_view,
        )
        generate_new_dataset_from_df(
            val_df,
            dataset_id,
            req.state.user_id,
            object_info,
            "validation",
            metadata_view,
        )

        return {"state": 0, "message": "generate new dataset successfully"}

    except Exception as error:  # noqa: BLE001 - preserve API failure envelope
        return {"state": 1, "message": str(error)}
