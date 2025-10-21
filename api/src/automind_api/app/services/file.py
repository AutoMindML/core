import hashlib
import tempfile

from fastapi import Response, status
from fastapi.responses import FileResponse
from pandas import DataFrame

from automind_api.db.connection import create_mssql_engine


def generate_file_response(source_file: DataFrame, res: Response):
    temp_source_file = tempfile.NamedTemporaryFile(delete=False, mode="w")

    try:
        source_file.to_csv(temp_source_file.name, index=False)
        return FileResponse(temp_source_file.name)
    finally:
        temp_source_file.close()
        res.status_code = status.HTTP_200_OK


def dataframe_to_mssql(source_file: DataFrame, user_id: int, target_id: int):
    mssql_engine = create_mssql_engine()
    source_file.to_sql(
        f"{user_id}_{target_id}", mssql_engine, "dbo", "replace", index=False
    )


def calculate_dataframe_md5(df: DataFrame) -> str:
    df_normalized = (
        df.sort_index()
        .sort_values(
            by=list(df.columns),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    df_string = df_normalized.to_csv(index=False).encode("utf-8")
    md5_hash = hashlib.md5(df_string).hexdigest()

    return md5_hash
