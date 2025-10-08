import hashlib
import uuid
from typing import Annotated

import sqlalchemy as sql
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import DBAPIError

from automind_api.db.connection import create_mssql_engine

from .utils import verify_member_id

router = APIRouter()


class AppPredictionAddRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    project_id: int
    model_id: int
    name: str
    des: str


def generate_api_key(id):
    raw_key = f"{id}-{uuid.uuid4()}"
    hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()
    return hashed_key


@router.post("/prediction")
def add_app(
    req: AppPredictionAddRequest,
    res: Response,
    mid: Annotated[int | None, Depends(verify_member_id)] = None,
):
    if mid is None:
        res.status_code = status.HTTP_401_UNAUTHORIZED
        return {"message": "session not found."}

    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        query = sql.text(
            """
            set nocount on;
            declare @new_id int;
            exec [dbo].[xp_add_app_prediction] @mid = :mid,
                @project_id = :project_id,
                @model_id = :model_id,
                @name = :name,
                @des = :des,
                @new_id = @new_id output;
            select @new_id as output;
        """
        )

        params = req.model_dump()
        params["mid"] = mid

        new_id = connection.execute(query, params).scalar()

        query = sql.text(
            """
            select api_key from [dbo].[vd_App_Prediction] where app_id = :app_id;
        """
        )

        api_key = connection.execute(query, {"app_id": new_id}).scalar()

        return {"new_id": new_id, "api_key": api_key}


class AppPredictionDeleteRequest(BaseModel):
    app_id: int


@router.delete("/prediction")
def delete_app(
    req: AppPredictionDeleteRequest,
    res: Response,
    mid: Annotated[int | None, Depends(verify_member_id)] = None,
):
    if mid is None:
        res.status_code = status.HTTP_401_UNAUTHORIZED
        return {"message": "session not found."}

    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                exec [dbo].[xp_delete_app_prediction] @mid = :mid, @app_id = :app_id;
                """
            )

            params = req.model_dump()
            params["mid"] = mid

            connection.execute(query, params)

            res.status_code = status.HTTP_200_OK

            return {"message": ""}

        except DBAPIError as e:
            res.status_code = status.HTTP_403_FORBIDDEN

            return {"message": e._sql_message()}
