from typing import Any, Mapping, Optional

import sqlalchemy as sql

from automind_api.app.models.i3s import ExecMutationSpOutput, ObjectInfo
from automind_api.db.connection import create_mssql_engine


def get_object_info(oid: int) -> Optional[ObjectInfo]:
    mssql = create_mssql_engine()

    with mssql.begin() as connection:
        query = sql.text("""
            select CName from dbo.[Object] where OID = :oid
        """)

        object = connection.execute(query, {"oid": oid}).fetchone()

        if object is None:
            return None

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
            declare @state int, @message nvarchar(4000);

            exec {sp_name}"""

        for k in opts.keys():
            query += f" @{k} = :{k},"

        query = (
            query
            + """ @state = @state output, @message = @message output;
            select @state, @message;
        """
        )

        row = connection.execute(sql.text(query), opts).fetchone()

        if row is None:
            return {"state": 1, "message": "internal server error"}

        result = row._tuple()
        return {"state": result[0], "message": result[1]}
