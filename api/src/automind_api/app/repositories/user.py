from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import sql
from sqlalchemy.exc import DBAPIError

from automind_api.app.models.app import ViewAppPrediction
from automind_api.app.models.view_sp import AvailableView
from automind_api.app.repositories.i3s import get_view_by_id
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


def verify_deployment_and_api_key(api_key: str, deployment_id: str):
    view_app_prediction: ViewAppPrediction = get_view_by_id(
        AvailableView.app_prediction,
        {"id": f"'{api_key}'", "id_col_name": "api_key"},
        ViewAppPrediction,
    )

    if not dict(view_app_prediction):
        raise HTTPException(status_code=401, detail="API key not valid")

    if view_app_prediction["deployment_id"] != deployment_id:
        raise HTTPException(status_code=401, detail="Deployment Id not valid")

    return view_app_prediction
