import json
from pathlib import Path

import mindsdb_sdk
from sqlalchemy import create_engine


def get_mindsdb_ml_engine_upload_url():
    f = open(f"{Path(__file__).parent.parent.parent.absolute()}/config.json")
    json_file = json.load(f)
    f.close()
    config = json_file["servert"]["mindsdb"]
    host = config["host"]
    port = config["port"]

    return f"http://{host}:{port}/api/handlers/byom"


def create_mssql_engine():
    f = open(f"{Path(__file__).parent.parent.parent.absolute()}/config.json")
    json_file = json.load(f)
    f.close()
    config = json_file["server"]["systemdb"]
    driver = config["driver"]
    user = config["user"]
    password = config["password"]
    host = config["host"]
    instance = config["instance"]
    port = config["port"]
    db = config["db"]
    url = f"mssql+pyodbc://{user}:{password}@{host}{instance}:{port}/{db}?driver={driver}&trustServerCertificate=yes"

    return create_engine(url, fast_executemany=False)


def connect_mindsdb_server():
    f = open(f"{Path(__file__).parent.parent.parent.absolute()}/config.json")
    json_file = json.load(f)
    f.close()
    config = json_file["server"]["mindsdb"]
    host = config["host"]
    port = config["port"]
    mindsdb_server = mindsdb_sdk.connect(f"http://{host}:{port}")

    return mindsdb_server
