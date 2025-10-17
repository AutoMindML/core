import json
from pathlib import Path


def get_config(
    config_key: str,
    config_name: str = "server",
):
    f = open(f"{Path().absolute()}/src/automind_api/configs/{config_name}.json")
    json_file = json.load(f)
    f.close()
    config = json_file[config_key]

    return config
