from sqlalchemy import sql

from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)


def get_data_source_file(oid: int, user_id: int):
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
            return None

        data_source = data_source._tuple()
        md5 = data_source[0]
        # source_type = data_source[1]

        if md5 not in [table.name for table in file_db.list_tables()]:
            return None

        source = file_db.get_table(md5)
        source_df = source.fetch()

        return source_df
