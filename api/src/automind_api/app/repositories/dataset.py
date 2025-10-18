from typing import Optional

from pandas import DataFrame
from sqlalchemy import sql

from automind_api.app.models.dataset import DatasetReturn, DatasetType
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)


def get_dataset(
    dataset_id: int, user_id: int, type: DatasetType, limit: int = 20
) -> Optional[DatasetReturn]:
    mssql_engine = create_mssql_engine()
    mindsdb_server = connect_mindsdb_server()

    match type:
        case "file":
            file_db = mindsdb_server.get_database("files")

            with mssql_engine.begin() as connection:
                query = sql.text(
                    """
                    select md5 from [dbo].[vd_Data_Source] where oid = :oid and owner_mid = :mid and source_type = 'file'
                    """
                )

                md5 = connection.execute(
                    query, {"mid": user_id, "oid": dataset_id}
                ).scalar()

                if (md5 is None) or (
                    md5 not in [table.name for table in file_db.list_tables()]
                ):
                    return None

                md5 = str(md5)
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
