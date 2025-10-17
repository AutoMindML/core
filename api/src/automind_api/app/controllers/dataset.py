import json
import tempfile
from io import StringIO
from typing import Annotated

import pandas as pd
import sqlalchemy as sql
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.exc import DBAPIError

from automind_api.app.models.dataset import AddDatabaseModel
from automind_api.app.services.user import verify_user
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)

dataset_router = APIRouter()


@dataset_router.post("/file")
async def add_data_source_file(
    name: Annotated[str, Form()],
    des: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    req: Request,
):
    user_id = req.state.user_id

    mssql_engine = create_mssql_engine()

    new_id = None
    file_content = await file.read()

    df = pd.read_csv(StringIO(file_content.decode()))
    df.to_sql(f"{user_id}_{name}", mssql_engine, "dbo", "replace")

    with mssql_engine.begin() as connection:
        params = {"mid": user_id, "name": name, "des": des}
        query = sql.text(
            """
            set nocount on;
            declare @new_id int;

            exec [dbo].[xp_add_data_source_file] @mid = :mid, @name = :name, @des = :des, @new_id = @new_id output;

            select @new_id as output;
            """
        )
        new_id = connection.execute(query, params).scalar()

        params = {"oid": new_id}
        query = sql.text(
            """
            select EName from Object where OID = :oid
        """
        )
        MD5 = connection.execute(query, params).scalar()

        mindsdb_server = connect_mindsdb_server()

        files_db = mindsdb_server.get_database("files")

        if str(MD5) not in [table.name for table in files_db.list_tables()]:
            files_db.create_table(str(MD5), df, True)

    return {
        "filename": file.filename,
        "file_size": len(file_content),
        "new_id": new_id,
    }


@dataset_router.post("/database")
def add_data_source_database(
    req: AddDatabaseModel,
    res: Response,
    user_id: Annotated[int | None, Depends(verify_user)],
):
    if user_id is None:
        res.status_code = status.HTTP_401_UNAUTHORIZED
        return {"message": "session not found."}

    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        params = req.model_dump()
        params["connection_args"] = json.dumps(params["connection_args"])
        params["mid"] = user_id

        query = sql.text(
            """
            set nocount on;
            declare @new_id int;

            exec [dbo].[xp_add_data_source_database] @mid = :mid,
                @name = :name, @des = :des, @engine = :engine,
                @connection_args = :connection_args, @new_id = @new_id output;

            select @new_id as output;
        """
        )
        new_id = connection.execute(query, params).scalar()

        params = {"oid": new_id}
        query = sql.text(
            """
            select EName from Object where OID = :oid
        """
        )
        MD5 = connection.execute(query, params).scalar()

        mindsdb_server = connect_mindsdb_server()

        if str(MD5) not in [
            database.name for database in mindsdb_server.list_databases()
        ]:
            mindsdb_server.create_database(
                engine=req.engine,
                name=str(MD5),
                connection_args=req.connection_args,
            )

    return {"engine": req.engine, "md5": MD5, "new_id": new_id}


class DeleteDataSouce(BaseModel):
    oid: int


@dataset_router.delete("/")
def delete_data_source(
    body: DeleteDataSouce,
    req: Request,
    res: Response,
):
    user_id = req.state.user_id

    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                exec [dbo].[xp_delete_data_source] @mid = :mid, @oid = :oid;
                """
            )

            params = body.model_dump()
            params["mid"] = user_id

            connection.execute(query, params)

            res.status_code = status.HTTP_200_OK

            return {"message": "delete data source successfully."}

        except DBAPIError as e:
            res.status_code = status.HTTP_403_FORBIDDEN

            return {"message": e._sql_message()}


@dataset_router.get("/")
def get_data_source_file(
    oid: int,
    req: Request,
    res: Response,
):
    user_id = req.state.user_id

    mssql_engine = create_mssql_engine()
    mindsdb_server = connect_mindsdb_server()
    file_db = mindsdb_server.get_database("files")

    with mssql_engine.begin() as connection:
        query = sql.text(
            """
            select md5, source_type from [dbo].[vd_Data_Source] where oid = :oid and owner_mid = :mid
            """
        )

        data_source = connection.execute(
            query, {"mid": user_id, "oid": oid}
        ).fetchone()

        if data_source is None:
            res.status_code = status.HTTP_404_NOT_FOUND
            return {"message": "data source is not exists."}

        data_source = data_source._tuple()
        md5 = data_source[0]
        # source_type = data_source[1]

        if md5 not in [table.name for table in file_db.list_tables()]:
            res.status_code = status.HTTP_404_NOT_FOUND
            return {
                "message": "can't not found table from given md5 in mindsdb files.",
            }

        source = file_db.get_table(md5)
        source_df = source.fetch()
        source_df: pd.DataFrame = source_df.iloc[:20]

        temp_source_file = tempfile.NamedTemporaryFile(delete=False, mode="w")

        try:
            source_df.to_csv(temp_source_file.name, index=False)
            return FileResponse(temp_source_file.name)
        finally:
            temp_source_file.close()
            res.status_code = status.HTTP_200_OK
