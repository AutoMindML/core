import sqlalchemy as sql
from fastapi import (
    APIRouter,
    Request,
    Response,
    status,
)
from sqlalchemy.exc import DBAPIError

from automind_api.app.models.project import (
    ProjectAddRequest,
    ProjectDeleteRequest,
)
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)

project_router = APIRouter()


@project_router.post("/")
def add_project(
    body: ProjectAddRequest,
    res: Response,
    req: Request,
):
    user_id = req.state.user_id
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

            params = body.model_dump()
            params["mid"] = user_id

            new_id = connection.execute(query, params).scalar()

            if new_id is not None:
                project_name = f"project_{new_id}"

                mindsdb_server = connect_mindsdb_server()

                if project_name not in [
                    project.name for project in mindsdb_server.list_projects() # pyright: ignore
                ]:
                    mindsdb_server.create_project(project_name) # pyright: ignore

            res.status_code = status.HTTP_200_OK

        except DBAPIError:
            res.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return {"newId": str(new_id)}


@project_router.delete("/")
def delete_project(
    body: ProjectDeleteRequest,
    req: Request,
    res: Response,
):
    user_id = req.state.user_id
    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                exec [dbo].[xp_delete_project] @mid = :mid, @cid = :cid;
                """
            )

            params = body.model_dump()
            params["mid"] = user_id

            connection.execute(query, params)

            if status == 0:
                project_name = f"project_{body.cid}"

                mindsdb_server = connect_mindsdb_server()

                if project_name in [
                    project.name for project in mindsdb_server.list_projects() # pyright: ignore
                ]:
                    mindsdb_server.drop_project(project_name) # pyright: ignore

            res.status_code = status.HTTP_200_OK
            return {"message": "delete project successfully"}

        except DBAPIError as e:
            res.status_code = status.HTTP_409_CONFLICT
            return {"message": e._sql_message()}
