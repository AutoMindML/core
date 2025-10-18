import json
import tempfile
from io import StringIO
from typing import Annotated

import pandas as pd
import sqlalchemy as sql
from fastapi import (
    APIRouter,
    File,
    Form,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.exc import DBAPIError

from automind_api.app.models.dataset import AddDatabaseBody, DatasetType
from automind_api.app.repositories.dataset import get_dataset
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)

dataset_router = APIRouter()


@dataset_router.post("/file")
async def add_data_source_file(
    req: Request,
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    des: Annotated[str, Form()] = "",
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
    req: Request,
    res: Response,
    body: AddDatabaseBody,
):
    user_id = req.state.user_id
    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        params = body.model_dump()
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
                engine=body.engine,
                name=str(MD5),
                connection_args=body.connection_args,
            )

    return {"engine": body.engine, "md5": MD5, "new_id": new_id}


@dataset_router.delete("/{dataset_id}")
def delete_data_source(
    dataset_id: int,
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

            params = {"mid": user_id, "oid": dataset_id}

            connection.execute(query, params)

            res.status_code = status.HTTP_200_OK

            return {"message": "delete data source successfully."}

        except DBAPIError as e:
            res.status_code = status.HTTP_403_FORBIDDEN

            return {"message": e._sql_message()}


@dataset_router.get("/{dataset_id}/preview/{rows}")
def get_dataset_preview(
    dataset_id: int,
    rows: int,
    req: Request,
    res: Response,
    dataset_type: DatasetType = "file",
):
    user_id = req.state.user_id
    dataset = get_dataset(
        dataset_id, user_id, dataset_type, rows if rows > 0 else -1
    )

    if dataset is None:
        res.status_code = status.HTTP_404_NOT_FOUND
        return {"message": "data source not exists"}

    temp_source_file = tempfile.NamedTemporaryFile(delete=False, mode="w")

    try:
        dataset["table"].to_csv(temp_source_file.name, index=False)
        return FileResponse(temp_source_file.name)
    finally:
        temp_source_file.close()
        res.status_code = status.HTTP_200_OK


@dataset_router.get("/{dataset_id}/columns")
def get_dataset_columns(
    dataset_id: int, req: Request, res: Response, dataset_type: DatasetType
):
    user_id = req.state.user_id
    dataset = get_dataset(dataset_id, user_id, dataset_type, 1)

    if dataset is None:
        res.status_code = status.HTTP_404_NOT_FOUND
        return {"message": "data source not exists"}

    return dataset["columns"]
