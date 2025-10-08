from typing import Dict, Optional

import pandas as pd
import sqlalchemy as sql
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict

from automind_api.db.connection import connect_mindsdb_server, create_mssql_engine

router = APIRouter()


async def verify_api_key(api_key: Optional[str] = Header(None)):
    mssql_engine = create_mssql_engine()

    with mssql_engine.begin() as connection:
        query = sql.text(
            """
            select * from [dbo].[vd_valid_api_keys]
        """
        )

        VALID_API_KEYS = [
            sequence[0] for sequence in connection.execute(query).fetchall()
        ]

        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return api_key


InputFeatures = list[Dict[str, float] | Dict[str, str]]


@router.post("/app")
async def use_app(req: InputFeatures, api_key: str = Depends(verify_api_key)):
    mssql_engine = create_mssql_engine()

    if len(req) == 0:
        return ORJSONResponse([])

    with mssql_engine.begin() as connection:
        params = {"api_key": api_key}
        query = sql.text(
            """
            select project_id, model_id, input_features, output_features from [dbo].[vd_Model] where api_key = :api_key;
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

            for req_input_features in req:
                if isinstance(input_features, str) and set(req_input_features) != set(
                    input_features
                ):
                    raise HTTPException(
                        400,
                        "Input features not correct. Require ("
                        + ", ".join(set(input_features))
                        + ") features",
                    )

                predicted_result = pd.DataFrame(model.predict(req_input_features))
                predicted_result = predicted_result.to_dict().get(output_feature)

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


class UseModel(BaseModel):
    model_id: int
    input_features: InputFeatures

    model_config = ConfigDict(protected_namespaces=())


@router.post("/model")
async def use_model(req: UseModel):
    mssql_engine = create_mssql_engine()

    if len(req.input_features) == 0:
        return ORJSONResponse([])

    with mssql_engine.begin() as connection:
        params = {"model_id": req.model_id}
        query = sql.text(
            """
            select project_id, model_id, input_features, output_features
            from [dbo].[vd_Model] where model_id = :model_id;
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

            for req_input_features in req.input_features:
                if isinstance(input_features, str) and set(req_input_features) != set(
                    input_features
                ):
                    raise HTTPException(
                        400,
                        "Input features not correct. Require ("
                        + ", ".join(set(input_features))
                        + ") features",
                    )

                predicted_result = pd.DataFrame(model.predict(req_input_features))
                predicted_result = predicted_result.to_dict().get(output_feature)

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
