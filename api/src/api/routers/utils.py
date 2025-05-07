from datetime import datetime
from typing import Annotated

from fastapi import Header
from sqlalchemy import sql
from sqlalchemy.exc import DBAPIError

from ...db.connection import create_mssql_engine


async def verify_member_id(mid: Annotated[int, Header()]):
    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                select 
                    top 1
                    [SID]
                    , ExpiredDT
                from MSession
                where MID = :mid
                order by [SID] desc;
                """
            )

            params = {"mid": mid}

            session = connection.execute(query, params).fetchone()

            if session is not None:
                expired_at: datetime = session[1]
                now = datetime.now()

                if now < expired_at:
                    return mid

            return None

        except DBAPIError:
            return None
