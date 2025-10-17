from datetime import datetime
from typing import Optional

from fastapi import Header, HTTPException
from sqlalchemy import sql
from sqlalchemy.exc import DBAPIError

from automind_api.db.connection import create_mssql_engine


async def get_session_id_by_user_id(user_id: int) -> Optional[int]:
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
                where [MID] = :mid
                order by [SID] desc;
                """
            )

            params = {"mid": user_id}

            session = connection.execute(query, params).fetchone()

            if session is not None:
                expired_at: datetime = session[1]
                now = datetime.now()

                if now < expired_at:
                    return session[0]

            return None

        except DBAPIError:
            return None


async def verify_api_key(api_key: Optional[str] = Header(None)):
    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        query = sql.text(
            """
            select * from [dbo].[vd_valid_api_keys]
        """
        )

        VALID_API_KEYS = [
            sequence[0] for sequence in connection.execute(query).fetchall()
        ]

        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return api_key
