import json
from typing import List

import sqlalchemy as sql
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse
from pandas import DataFrame, concat
from sqlalchemy.exc import DBAPIError
from starlette.status import HTTP_404_NOT_FOUND

from automind_api.app.models.app import (
    CreateAppBody,
    CreateAppParameter,
    DeleteAppBody,
)
from automind_api.app.models.model import ModelPredictionServiceBody
from automind_api.app.models.view_sp import (
    AvailableSP,
    AvailableView,
    ViewModel,
)
from automind_api.app.repositories.i3s import exec_mutation_sp, get_view_by_id
from automind_api.app.repositories.user import verify_deployment_and_api_key
from automind_api.db.connection import (
    connect_mindsdb_server,
    create_mssql_engine,
)

app_router = APIRouter()


@app_router.post("/")
def create_app(
    body: CreateAppBody,
    req: Request,
):
    user_id = req.state.user_id
    params: CreateAppParameter = {
        "user_id": user_id,
        "project_id": body.project_id,
        "name": body.name,
        "des": body.des,
    }

    return exec_mutation_sp(AvailableSP.create_app_prediction, params)


@app_router.post("/deployment/{deployment_id}")
def app_prediction(
    deployment_id: str,
    body: ModelPredictionServiceBody,
    req: Request,
    res: Response,
):
    view_app_prediction = verify_deployment_and_api_key(
        req.headers.get("X-API-Key", ""), deployment_id
    )
    view_model: ViewModel = get_view_by_id(
        AvailableView.model,
        {"id": body.model_id, "id_col_name": "model_id"},
        ViewModel,
    )

    if not dict(view_model):
        raise HTTPException(status_code=401, detail=" Model id not valid")

    if view_app_prediction["project_id"] != view_model["project_id"]:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail="Source not found"
        )

    dataset = DataFrame(body.input)
    project_name = f"project_{view_model['project_id']}"
    model_name = f"model_{view_model['model_id']}"

    mindsdb_server = connect_mindsdb_server()
    project = mindsdb_server.projects.get(project_name)  # pyright: ignore
    model = project.models.get(model_name)
    model_status = model.get_status()

    if model_status == "complete":
        output_features: List[str] = list(
            json.loads(view_model["output_features"])
        )
        input_features: List[str] = list(
            json.loads(view_model["input_features"])
        )

        source_df = dataset.copy()

        y = source_df[output_features]
        X = source_df.drop(columns=output_features)

        # check if input features correct
        if set(X.columns.tolist()) != set(input_features):
            res.status_code = status.HTTP_400_BAD_REQUEST
            return None

        pred_df = DataFrame(model.predict(X.fillna(0)))
        result_df = concat([y, pred_df["prediction"], X], axis=1)

        if body.limit > 0:
            result_df = result_df.iloc[: body.limit]

        # return generate_file_response(result_df, res)
        return ORJSONResponse(result_df.to_dict(orient="records"))

    return None


@app_router.delete("/")
def delete_app(
    body: DeleteAppBody,
    req: Request,
    res: Response,
):
    user_id = req.state.user_id
    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        try:
            query = sql.text(
                """
                exec [dbo].[xp_delete_app_prediction] @mid = :mid, @app_id = :app_id;
                """
            )

            params = body.model_dump()
            params["mid"] = user_id

            connection.execute(query, params)

            res.status_code = status.HTTP_200_OK

            return {"message": ""}

        except DBAPIError as e:
            res.status_code = status.HTTP_403_FORBIDDEN

            return {"message": e._sql_message()}
