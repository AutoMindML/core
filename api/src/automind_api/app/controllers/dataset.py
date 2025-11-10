import json
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

from automind_api.app.models.dataset import (
    AddDatabaseBody,
    AddDatasetFileParameter,
    DatasetType,
    DeleteDatasetParameter,
    dtype_map,
)
from automind_api.app.repositories.dataset import (
    dataset_to_mindsdb,
    get_dataset,
)
from automind_api.app.repositories.i3s import exec_mutation_sp
from automind_api.app.services.file import (
    calculate_dataframe_md5,
    generate_file_response,
)
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
    file_content = await file.read()
    df = pd.read_csv(StringIO(file_content.decode()))
    md5 = calculate_dataframe_md5(df).upper()
    rows, cols = df.shape
    new_column_mapping = {
        col: col.replace("(", "_").replace(")", "").replace(".", "_")
        for col in df.columns
    }
    df.rename(columns=new_column_mapping, inplace=True)
    col_names = df.columns.to_list()
    col_types = [
        dtype_map.get(str(dt), str(dt)) for col, dt in df.dtypes.items()
    ]
    opts: AddDatasetFileParameter = {
        "user_id": req.state.user_id,
        "md5": md5,
        "rows": rows,
        "cols": cols,
        "col_names": ",".join(col_names),
        "col_types": ",".join(col_types),
        "name": name,
        "des": des,
        "size": float(df.memory_usage(index=False, deep=True).sum()),
        "size_unit": "bytes",
        "quality": 0.0,
    }
    result = exec_mutation_sp("[dbo].[xp_add_data_source_file]", opts)
    dataset_to_mindsdb(df, md5)

    return result


@dataset_router.post("/database")
def add_data_source_database(
    req: Request,
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
            database.name
            for database in mindsdb_server.list_databases()  # pyright: ignore
        ]:
            mindsdb_server.create_database(  # pyright: ignore
                engine=body.engine,
                name=str(MD5),
                connection_args=body.connection_args,
            )

    return {"engine": body.engine, "md5": MD5, "new_id": new_id}


@dataset_router.delete("/{dataset_id}")
def delete_data_source(
    dataset_id: int,
    req: Request,
):
    params: DeleteDatasetParameter = {
        "user_id": req.state.user_id,
        "dataset_id": dataset_id,
    }

    return exec_mutation_sp("[dbo].[xp_delete_data_source]", params)


@dataset_router.get("/{dataset_id}/preview")
def get_dataset_preview(
    dataset_id: int,
    req: Request,
    res: Response,
    rows: int = 20,
    dataset_type: DatasetType = "file",
):
    user_id = req.state.user_id
    dataset = get_dataset(dataset_id, user_id, dataset_type, rows)

    if dataset is None:
        res.status_code = status.HTTP_404_NOT_FOUND
        return {"message": "data source not exists"}

    return generate_file_response(dataset["table"], res)


@dataset_router.get("/{dataset_id}/columns")
def get_dataset_columns(
    dataset_id: int,
    req: Request,
    res: Response,
    dataset_type: DatasetType = "file",
):
    user_id = req.state.user_id
    dataset = get_dataset(dataset_id, user_id, dataset_type, 1)

    if dataset is None:
        res.status_code = status.HTTP_404_NOT_FOUND
        return {"message": "data source not exists"}

    return dataset["columns"]
