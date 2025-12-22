import hashlib
import tempfile

from fastapi import Response, status
from fastapi.responses import FileResponse
from pandas import DataFrame

from automind_api.app.models.dataset import (
    DatasetCommonInfo,
    dtype_map,
)
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

    return md5_hash.upper()


def generate_dataframe_info_by_ref(df: DataFrame) -> DatasetCommonInfo:
    new_column_mapping = {
        col: col.replace("(", "_").replace(")", "").replace(".", "_")
        for col in df.columns
    }
    df.rename(columns=new_column_mapping, inplace=True)
    md5 = calculate_dataframe_md5(df).upper()
    rows, cols = df.shape
    new_column_mapping = {
        col: col.replace("(", "_").replace(")", "").replace(".", "_")
        for col in df.columns
    }
    df.rename(columns=new_column_mapping, inplace=True)
    col_names = df.columns.to_list()
    col_types = [dtype_map.get(str(dt), str(dt)) for _, dt in df.dtypes.items()]

    return {
        "md5": md5,
        "rows": rows,
        "cols": cols,
        "col_names": ",".join(col_names),
        "col_types": ",".join(col_types),
        "size": float(df.memory_usage(index=False, deep=True).sum()),
        "size_unit": "bytes",
        "quality": 0.0,
    }
