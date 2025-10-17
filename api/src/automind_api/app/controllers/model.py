from time import sleep

import pandas as pd
import sqlalchemy as sql
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse, ORJSONResponse
from sqlalchemy.exc import DBAPIError
from starlette.status import HTTP_404_NOT_FOUND

from automind_api.app.models.model import (
    ModelAddRequest,
    ModelDeleteRequest,
    ModelPredictionBody,
)
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)

model_router = APIRouter()


def update_model(project_id: int, model_id: int):
    mindsdb_server = connect_mindsdb_server()
    mssql_engine = create_mssql_engine()

    project_name = f"project_{project_id}"
    model_name = f"model_{model_id}"

    if project_name in [
        project.name for project in mindsdb_server.list_projects()
    ]:
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

                        if (model_inputs is not None) and (
                            model_outputs is not None
                        ):
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


@model_router.post("/train")
def add_model(
    body: ModelAddRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    res: Response,
):
    user_id = req.state.user_id
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

        params = body.model_dump()
        params["mid"] = user_id

        new_id = connection.execute(query, params).scalar()

        if new_id is None:
            raise HTTPException(500, "Failed to create model object.")

        mindsdb_server = connect_mindsdb_server()

        project_name = f"project_{body.cid}"

        if project_name not in [
            project.name for project in mindsdb_server.list_projects()
        ]:
            return JSONResponse(None, HTTP_404_NOT_FOUND)

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
                body.predict,
                engine_md5,
                select_data_query,
                "files",
            )
        else:
            project.create_model(
                model_name,
                body.predict,
                engine_md5,
                body.select_data_query,
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
                    "input": str(model_inputs[0]),
                    "output": str(model_outputs[0]),
                }
                query = sql.text(
                    """
                    exec [dbo].[xp_set_model_info] @model_id = :model_id, @input = :input, @output = :output;
                """
                )

                connection.execute(query, params)

        background_tasks.add_task(update_model, body.cid, new_id)

    return {
        "status": 0,
        "message": "model has been created properly.",
        "new_id": new_id,
    }


@model_router.delete("/")
def delete_model(
    body: ModelDeleteRequest,
    req: Request,
    res: Response,
):
    user_id = req.state.user_id
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

            params = body.model_dump()
            params["mid"] = user_id

            connection.execute(query, params)

            if status == 0:
                project_name = f"project_{body.project_id}"

                mindsdb_server = connect_mindsdb_server()

                if project_name in [
                    project.name for project in mindsdb_server.list_projects()
                ]:
                    project = mindsdb_server.get_project(project_name)
                    model_name = f"model_{body.model_id}"

                    if model_name in [
                        model.name for model in project.list_models()
                    ]:
                        project.drop_model(model_name)

            res.status_code = status.HTTP_200_OK

            return {"message": ""}

        except DBAPIError as e:
            res.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

            return {"message": e._sql_message()}


@model_router.post("/predict")
async def use_model(body: ModelPredictionBody, req: Request):
    user_id = req.state.user_id
    mssql_engine = create_mssql_engine()

    if len(body.input_features) == 0:
        return ORJSONResponse([])

    with mssql_engine.begin() as connection:
        params = {"model_id": body.model_id, "mid": user_id}
        query = sql.text(
            """
            select project_id, model_id, input_features, output_features
            from [dbo].[vd_Model] where model_id = :model_id and owner_mid = :mid;
            """
        )

        models = [
            (sequence[0], sequence[1], sequence[2], sequence[3])
            for sequence in connection.execute(query, params).fetchall()
        ]

        input_features = str(models[0][2]).split(",")
        output_features = str(models[0][3]).split(",")

        project_id = models[0][0]
        model_id = models[0][1]
        project_name = f"project_{project_id}"
        model_name = f"model_{model_id}"

        mindsdb_server = connect_mindsdb_server()
        project = mindsdb_server.get_project(project_name)
        model = project.get_model(model_name)
        model_status = model.get_status()

        if model_status == "complete":
            model_outputs = []
            output_feature = output_features[0]

            for req_input_features in body.input_features:
                if isinstance(input_features, str) and set(
                    req_input_features
                ) != set(input_features):
                    raise HTTPException(
                        400,
                        "Input features not correct. Require ("
                        + ", ".join(set(input_features))
                        + ") features",
                    )

                predicted_result = pd.DataFrame(
                    model.predict(req_input_features)
                )
                predicted_result = predicted_result.to_dict().get(
                    output_feature
                )

                if predicted_result is not None:
                    model_output = predicted_result.get(0)
                    model_outputs.append({output_feature: model_output})
                else:
                    model_outputs.append({output_feature: None})

            return ORJSONResponse(model_outputs)

        else:
            return {
                "model_status": model_status,
                "message": "model can't be used currently.",
            }
