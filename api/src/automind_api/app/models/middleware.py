from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from automind_api.app.repositories.user import get_session_id_by_user_id
from automind_api.configs import get_config


class HeaderSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "X-User-Id"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        config = get_config("api")
        url = f"http://{config['host']}:{config['port']}"

        fastapi_urls = [
            f"{url}/docs",
            f"{url}/openapi.json",
            f"{url}/.well-known/appspecific/com.chrome.devtools.json",
        ]

        if (request.url._url in fastapi_urls) or (config["env"] == "test"):
            request.state.user_id = 1
            request.state.session_id = 1
            response = await call_next(request)
            return response

        user_id = request.headers.get(self.header_name)

        if (user_id is None) or (user_id == ""):
            raise HTTPException(
                status_code=401,
                detail="user id not found in headers.",
            )

        session_id = await get_session_id_by_user_id(int(user_id))

        if session_id is None:
            raise HTTPException(
                status_code=401, detail="session not found or expired."
            )

        request.state.user_id = int(user_id)
        request.state.session_id = int(session_id)

        response: Response = await call_next(request)
        return response
