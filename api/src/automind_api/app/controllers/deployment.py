from fastapi import APIRouter

deployment_router = APIRouter()


@deployment_router.post("/{deployment_id}")
def deployment():
    pass
