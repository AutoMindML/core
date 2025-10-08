from contextlib import asynccontextmanager

import requests
import sqlalchemy as sql
from fastapi import FastAPI

from .db.connection import connect_mindsdb_server, create_mssql_engine
from .routers.app import router as app_router
from .routers.data import router as data_router
from .routers.model import router as model_router
from .routers.project import router as project_router
from .routers.service import router as service_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mssql_engine = create_mssql_engine()
        with mssql_engine.begin() as connection:
            connection.execute(sql.text("select 1"))
    except sql.exc.SQLAlchemyError as e:
        print(f"connect to mssql failed, reason: {e}")

    try:
        connect_mindsdb_server()
    except requests.HTTPError as e:
        print(f"connect to mindsdb failed, reason: {e}")

    yield


app = FastAPI(lifespan=lifespan)

app.include_router(project_router, prefix="/api")
app.include_router(data_router, prefix="/api/data")
app.include_router(model_router, prefix="/api/model")
app.include_router(app_router, prefix="/api/app")
app.include_router(service_router, prefix="/api/service")
