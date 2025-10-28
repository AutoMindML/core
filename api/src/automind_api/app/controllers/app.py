import sqlalchemy as sql
from fastapi import APIRouter, Request, Response, status
from sqlalchemy.exc import DBAPIError

from automind_api.app.models.app import AddAppBody, DeleteAppBody
from automind_api.db.connection import create_mssql_engine

app_router = APIRouter()


@app_router.post("/")
def add_app(
    body: AddAppBody,
    req: Request,
):
    user_id = req.state.user_id
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

        params = body.model_dump()
        params["mid"] = user_id

        new_id = connection.execute(query, params).scalar()

        query = sql.text(
            """
            select api_key from [dbo].[vd_App_Prediction] where app_id = :app_id;
        """
        )

        api_key = connection.execute(query, {"app_id": new_id}).scalar()

        return {"new_id": new_id, "api_key": api_key}


@app_router.delete("/")
def delete_app(
    body: DeleteAppBody,
    req: Request,
    res: Response,
):
    user_id = req.state.user_id
    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                exec [dbo].[xp_delete_app_prediction] @mid = :mid, @app_id = :app_id;
                """
            )

            params = body.model_dump()
            params["mid"] = user_id

            connection.execute(query, params)

            res.status_code = status.HTTP_200_OK

            return {"message": ""}

        except DBAPIError as e:
            res.status_code = status.HTTP_403_FORBIDDEN

            return {"message": e._sql_message()}
