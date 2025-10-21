from typing import Any, Mapping, TypeVar

import sqlalchemy as sql
from sqlalchemy.exc import ProgrammingError

from automind_api.app.models.i3s import (
    ExecMutationSpOutput,
    GetViewByIdOpts,
    ObjectInfo,
)
from automind_api.db.connection import create_mssql_engine

T = TypeVar("T")


def get_object_info(oid: int) -> ObjectInfo:
    mssql = create_mssql_engine()

    with mssql.begin() as connection:
        query = sql.text("""
            select CName from dbo.[Object] where OID = :oid
        """)

        object = connection.execute(query, {"oid": oid}).fetchone()

        if object is None:
            return {"CName": ""}

        row = object._tuple()

        return {"CName": row[0]}


def exec_mutation_sp(
    sp_name: str, opts: Mapping[str, Any]
) -> ExecMutationSpOutput:
    """
    Notice: this method opts key name must the same as mssql sp params name
    """
    mssql = create_mssql_engine()

    with mssql.begin() as connection:
        query = f"""
            set nocount on;
            declare 
                @state int
                , @message nvarchar(4000)
                , @new_id int;

            exec {sp_name}"""

        for k in opts.keys():
            query += f" @{k} = :{k},"

        query = (
            query
            + """
                @state = @state output
                , @message = @message output
                , @new_id = @new_id output;
            select @state, @message, @new_id;
        """
        )

        try:
            row = connection.execute(sql.text(query), opts).fetchone()

            if row is None:
                return {
                    "state": 1,
                    "message": "internal server error",
                    "new_id": -1,
                }

            result = row._tuple()
            return {
                "state": result[0],
                "message": result[1],
                "new_id": result[2],
            }
        except ProgrammingError as e:
            return {"state": 2, "message": e._message(), "new_id": None}


def get_view_by_id(view_name: str, opts: GetViewByIdOpts, schema) -> Any:
    """
    Notice: this method key name must the same as mssql view column names
    """
    mssql = create_mssql_engine()
    keys = list(schema.__annotations__)

    with mssql.begin() as connection:
        query = "select"

        for k in keys:
            query += f" {k},"

        query = (
            query.rstrip(",")
            + f" from {view_name} where {opts['id_col_name']} = {opts['id']}"
        )
        row = connection.execute(sql.text(query)).fetchone()

        if row is None:
            return {}

        return row._mapping
