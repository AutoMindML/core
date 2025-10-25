from fastapi import Response
from starlette import status

from automind_api.app.models.dataset import VerifyDatasetResponse
from automind_api.app.repositories.dataset import get_dataset


def verify_dataset(
    dataset_id: int, user_id: int, res: Response, limit: int = 20,
) -> VerifyDatasetResponse:
    dataset = get_dataset(
        dataset_id,
        user_id,
        limit=limit
    )

    if dataset is None:
        res.status_code = status.HTTP_404_NOT_FOUND
        return {
            "state": 2,
            "message": "dataset is not exists or user has no permission",
            "table": None,
            "columns": None,
        }

    return {"state": 0, "message": "dataset is ready", **dataset}
