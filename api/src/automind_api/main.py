from contextlib import asynccontextmanager

import requests
import sqlalchemy as sql
from fastapi import FastAPI

from automind_api.app.controllers.app import app_router
from automind_api.app.controllers.data_fusion import data_fusion_router
from automind_api.app.controllers.dataset import dataset_router
from automind_api.app.controllers.model import model_router
from automind_api.app.controllers.project import project_router
from automind_api.app.models.middleware import HeaderSessionMiddleware
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
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

app.add_middleware(HeaderSessionMiddleware)
app.include_router(dataset_router, prefix="/api/dataset")
app.include_router(project_router, prefix="/api/project")
app.include_router(model_router, prefix="/api/automl/model")
app.include_router(data_fusion_router, prefix="/api/automl/data-fusion")
app.include_router(app_router, prefix="/api/app")
