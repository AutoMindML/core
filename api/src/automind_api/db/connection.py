import json
from pathlib import Path

from mindsdb_sdk.connect import connect
from mindsdb_sdk.server import Server
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_mindsdb_ml_engine_upload_url():
    f = open(f"{Path().absolute()}/src/automind_api/configs/server.json")
    json_file = json.load(f)
    f.close()
    config = json_file["mindsdb"]
    host = config["host"]
    port = config["port"]

    return f"http://{host}:{port}/api/handlers/byom"


def create_mssql_engine() -> Engine:
    f = open(f"{Path().absolute()}/src/automind_api/configs/server.json")
    json_file = json.load(f)
    f.close()

    config = json_file["systemdb"]
    driver = config["driver"]
    user = config["user"]
    password = config["password"]
    host = config["host"]
    instance = config["instance"]
    port = config["port"]
    db = config["db"]
    url = f"mssql+pyodbc://{user}:{password}@{host}{instance}:{port}/{db}?driver={driver}&trustServerCertificate=yes"

    engine = create_engine(url, fast_executemany=False)

    return engine


# version history   : https://pypi.org/project/mindsdb-sdk/#history
# repo              : https://github.com/mindsdb/mindsdb_python_sdk/blob/main/mindsdb_sdk/server.py
# documents         : https://mindsdb.github.io/mindsdb_python_sdk/
def connect_mindsdb_server() -> Server:
    f = open(f"{Path().absolute()}/src/automind_api/configs/server.json")
    json_file = json.load(f)
    f.close()
    config = json_file["mindsdb"]
    host = config["host"]
    port = config["port"]
    mindsdb_server = connect(f"http://{host}:{port}")

    return mindsdb_server
