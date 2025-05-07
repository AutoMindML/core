from typing import Annotated

import sqlalchemy as sql
from fastapi import (
    APIRouter,
    Depends,
    Response,
    status,
)
from pydantic import BaseModel
from sqlalchemy.exc import DBAPIError

from ...db.connection import connect_mindsdb_server, create_mssql_engine
from .utils import verify_member_id

router = APIRouter()


class ProjectAddRequest(BaseModel):
    name: str
    des: str


@router.post("/project")
def add_project(
    req: ProjectAddRequest,
    res: Response,
    mid: Annotated[int | None, Depends(verify_member_id)] = None,
):
    if mid is None:
        res.status_code = status.HTTP_401_UNAUTHORIZED
        return {"message": "session not found."}

    mssql_engine = create_mssql_engine()

    new_id = None

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                set nocount on;
                declare @new_id int;
                exec [dbo].[xp_add_project] @mid = :mid, @name = :name, @des = :des, @new_id = @new_id output;
                select @new_id as output;
            """
            )

            params = req.model_dump()
            params["mid"] = mid

            new_id = connection.execute(query, params).scalar()

            if new_id is not None:
                project_name = f"project_{new_id}"

                mindsdb_server = connect_mindsdb_server()

                if project_name not in [
                    project.name for project in mindsdb_server.list_projects()
                ]:
                    mindsdb_server.create_project(project_name)

            res.status_code = status.HTTP_200_OK

        except DBAPIError:
            res.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return {"new_id": new_id}


class ProjectDeleteRequest(BaseModel):
    cid: int


@router.delete("/project")
def delete_project(
    req: ProjectDeleteRequest,
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
                exec [dbo].[xp_delete_project] @mid = :mid, @cid = :cid;
                """
            )

            params = req.model_dump()
            params["mid"] = mid

            connection.execute(query, params)

            if status == 0:
                project_name = f"project_{req.cid}"

                mindsdb_server = connect_mindsdb_server()

                if project_name in [
                    project.name for project in mindsdb_server.list_projects()
                ]:
                    mindsdb_server.drop_project(project_name)

            res.status_code = status.HTTP_200_OK
            return {"message": "delete project successfully"}

        except DBAPIError as e:
            res.status_code = status.HTTP_409_CONFLICT
            return {"message": e._sql_message()}
