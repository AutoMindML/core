from typing import Optional

from pandas import DataFrame
from sqlalchemy import sql

from automind_api.app.models.dataset import (
    DatasetReturn,
    DatasetType,
    ViewDataSource,
)
from automind_api.app.models.view_sp import AvailableView
from automind_api.app.repositories.i3s import get_view_by_id
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)


def get_dataset(
    dataset_id: int,
    user_id: int,
    type: Optional[DatasetType] = None,
    limit: int = 20,
) -> Optional[DatasetReturn]:
    mssql_engine = create_mssql_engine()
    mindsdb_server = connect_mindsdb_server()

    if type is None:
        view: ViewDataSource = get_view_by_id(
            AvailableView.dataset,
            {"id": dataset_id, "id_col_name": "oid"},
            ViewDataSource,
        )
        type = view.get("source_type")

    match type:
        case "file" | "fusion":
            file_db = mindsdb_server.get_database("files")  # pyright: ignore[reportAttributeAccessIssue]

            with mssql_engine.begin() as connection:
                query = sql.text(
                    """
                    select md5 from [dbo].[vd_Data_Source] where oid = :oid and owner_mid = :mid and source_type in ('file', 'fusion')
                    """
                )

                md5 = connection.execute(
                    query, {"mid": user_id, "oid": dataset_id}
                ).scalar()

                if md5 is not None:
                    md5 = str(md5).lower()
                else:
                    return None

                if md5 not in [table.name for table in file_db.list_tables()]:
                    md5 = md5.upper()

                table = file_db.get_table(md5)

                if limit > 0:
                    table = table.limit(limit)

                source_df = DataFrame(table.fetch())

                return {
                    "table": source_df,
                    "columns": source_df.columns.to_list(),
                }
        case _:
            return None


def dataset_to_mindsdb(dataset: DataFrame, md5: str):
    md5 = md5.lower()
    mindsdb_server = connect_mindsdb_server()
    files_db = mindsdb_server.get_database("files")  # pyright: ignore[reportAttributeAccessIssue]
    file_md5s = [table.name for table in files_db.list_tables()]
    if (md5 not in file_md5s) and (md5.upper() not in file_md5s):
        files_db.create_table(md5, dataset, True)
