from automind_api.app.models.i3s import BasePostResponse


def generate_common_response(state: int, message: str) -> BasePostResponse:
    return {"state": state, "message": message}
