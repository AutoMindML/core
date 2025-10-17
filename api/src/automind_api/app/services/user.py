from typing import Annotated

from fastapi import Header

from automind_api.app.repositories.user import get_session_id_by_user_id


async def verify_user(session_id: Annotated[int, Header()]):
    return get_session_id_by_user_id(session_id)
