from time import sleep
from typing import Annotated

import pandas as pd
import sqlalchemy as sql
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import DBAPIError

from ...db.connection import connect_mindsdb_server, create_mssql_engine
from .utils import verify_member_id

router = APIRouter()


class ModelAddRequest(BaseModel):
    cid: int
    data_oid: int
    engine_oid: int
    name: str
    des: str
    predict: str
    tag: str
    select_data_query: str
    training_options: str


def update_model(project_id: int, model_id: int):
    mindsdb_server = connect_mindsdb_server()
    mssql_engine = create_mssql_engine()

    project_name = f"project_{project_id}"
    model_name = f"model_{model_id}"

    if project_name in [project.name for project in mindsdb_server.list_projects()]:
        project = mindsdb_server.get_project(project_name)

        if model_name in [model.name for model in project.list_models()]:
            while True:
                model_info = (
                    mindsdb_server.query(
                        f"select * from information_schema.models where Name = '{model_name}'"
                    )
                    .fetch()
                    .iloc[0]
                )

                with mssql_engine.begin() as connection:
                    params = {
                        "model_id": model_id,
                        "select_data_query": model_info["SELECT_DATA_QUERY"],
                        "active": 1 if model_info["ACTIVE"] else 0,
                        "status": model_info["STATUS"],
                        "accuracy": model_info["ACCURACY"],
                        "training_time": model_info["TRAINING_TIME"],
                        "update_status": model_info["UPDATE_STATUS"],
                        "error": model_info["ERROR"],
                        "current_training_phase": int(
                            model_info["CURRENT_TRAINING_PHASE"]
                        )
                        if model_info["CURRENT_TRAINING_PHASE"]
                        else 0,
                        "total_training_phases": int(
                            model_info["TOTAL_TRAINING_PHASES"]
                        )
                        if model_info["TOTAL_TRAINING_PHASES"]
                        else 0,
                        "training_options": model_info["TRAINING_OPTIONS"],
                    }

                    query = sql.text(
                        """
                            exec [dbo].[xp_update_model] @model_id = :model_id,
                                @select_data_query = :select_data_query,
                                @active = :active,
                                @status = :status,
                                @accuracy = :accuracy,
                                @training_time = :training_time,
                                @update_status = :update_status,
                                @error = :error,
                                @current_training_phase = :current_training_phase,
                                @total_training_phases = :total_training_phases,
                                @training_options = :training_options;
                        """
                    )

                    connection.execute(query, params)

                if params["status"] == "complete":
                    with mssql_engine.begin() as connection:
                        model = project.get_model(model_name)
                        model_info = pd.DataFrame(model.describe("info"))
                        model_inputs = model_info.get("inputs")
                        model_outputs = model_info.get("outputs")

                        if (model_inputs is not None) and (model_outputs is not None):
                            params = {
                                "model_id": model_id,
                                "input": str(",".join(model_inputs[0])),
                                "output": str(",".join(model_outputs[0])),
                            }
                            query = sql.text(
                                """
                                exec [dbo].[xp_set_model_info] @model_id = :model_id,
                                    @input = :input, @output = :output;
                                """
                            )

                            connection.execute(query, params)
                    break

                elif params["status"] == "error":
                    break

                sleep(5)


@router.post("/")
def add_model(
    req: ModelAddRequest,
    background_tasks: BackgroundTasks,
    res: Response,
    mid: Annotated[int | None, Depends(verify_member_id)] = None,
):
    if mid is None:
        res.status_code = status.HTTP_401_UNAUTHORIZED
        return {"message": "session not found."}

    mssql_engine = create_mssql_engine()

    new_id = None

    with mssql_engine.begin() as connection:
        query = sql.text(
            """
            set nocount on;
            declare @new_id int;
            exec [dbo].[xp_add_model] @mid = :mid,
                @cid = :cid,
                @data_oid = :data_oid,
                @engine_oid = :engine_oid,
                @name = :name, @des = :des,
                @predict = :predict,
                @tag = :tag,
                @new_id = @new_id output;
            select @new_id as output;
        """
        )

        params = req.model_dump()
        params["mid"] = mid

        new_id = connection.execute(query, params).scalar()

        if new_id is None:
            raise HTTPException(500, "Failed to create model object.")

        mindsdb_server = connect_mindsdb_server()

        project_name = f"project_{req.cid}"

        if project_name in [project.name for project in mindsdb_server.list_projects()]:
            project = mindsdb_server.get_project(project_name)
            model_name = f"model_{new_id}"

            params = {"model_id": new_id}
            query = sql.text(
                """
                select data_source_md5, engine_md5, data_source_type from vd_Model where model_id = :model_id;
            """
            )

            model_object = connection.execute(query, params).fetchone()

            if model_object is None:
                raise HTTPException(
                    500,
                    "Failed to create model. reason: can't find model id from vd_Model.",
                )

            model_object = model_object._tuple()
            data_source_md5 = model_object[0]
            engine_md5 = model_object[1]
            data_source_type = model_object[2]

            if data_source_type == "file":
                select_data_query = f"""
                    select * from {data_source_md5}
                """
                project.create_model(
                    model_name,
                    req.predict,
                    engine_md5,
                    select_data_query,
                    "files",
                )
            else:
                project.create_model(
                    model_name,
                    req.predict,
                    engine_md5,
                    req.select_data_query,
                    data_source_md5,
                )

            if model_name in [model.name for model in project.list_models()]:
                model = project.get_model(model_name)
                model_info = pd.DataFrame(model.describe("info"))
                model_inputs = model_info.get("inputs")
                model_outputs = model_info.get("outputs")

                if (model_inputs is not None) and (model_outputs is not None):
                    params = {
                        "model_id": new_id,
                        "input": model_inputs[0],
                        "output": model_outputs[0],
                    }
                    query = sql.text(
                        """
                        exec [dbo].[xp_set_model_info] @model_id = :model_id, @input = :input, @output = :output;
                    """
                    )

                    connection.execute(query, params)

            background_tasks.add_task(update_model, req.cid, new_id)

    return {
        "status": 0,
        "message": "model has been created properly.",
        "new_id": new_id,
    }


class ModelDeleteRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    project_id: int
    model_id: int

    model_config = ConfigDict(protected_namespaces=())


@router.delete("/")
def delete_model(
    req: ModelDeleteRequest,
    res: Response,
    mid: Annotated[int | None, Depends(verify_member_id)] = None,
):
    if mid is None:
        res.status_code = status.HTTP_401_UNAUTHORIZED
        return {"message": "session not found."}

    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                exec [dbo].[xp_delete_model] @mid = :mid,
                    @project_id = :project_id,
                    @model_id = :model_id;
                """
            )

            params = req.model_dump()
            params["mid"] = mid

            connection.execute(query, params)

            if status == 0:
                project_name = f"project_{req.project_id}"

                mindsdb_server = connect_mindsdb_server()

                if project_name in [
                    project.name for project in mindsdb_server.list_projects()
                ]:
                    project = mindsdb_server.get_project(project_name)
                    model_name = f"model_{req.model_id}"

                    if model_name in [model.name for model in project.list_models()]:
                        project.drop_model(model_name)

            res.status_code = status.HTTP_200_OK

            return {"message": ""}

        except DBAPIError as e:
            res.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

            return {"message": e._sql_message()}
