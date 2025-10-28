from fastapi import Response
from starlette import status


def generate_409_conflict_response(res: Response):
    res.status_code = status.HTTP_409_CONFLICT
    return {
        "state": 2,
        "message": "dataset is not found or target column not be specified or llm response not exists",
    }
