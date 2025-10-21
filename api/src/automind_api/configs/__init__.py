import json
from pathlib import Path
from typing import Any


def get_config(
    config_key: str,
    config_name: str = "server",
) -> dict[str, Any]:
    f = open(f"{Path().absolute()}/src/automind_api/configs/{config_name}.json")
    json_file = json.load(f)
    f.close()
    config = json_file[config_key]

    return config
