import json
from pathlib import Path
from typing import Any

logger_name = "automind-core"


def get_config(
    config_key: str,
    config_name: str = "public",
) -> dict[str, Any]:
    f = open(Path().parent.absolute() / f"{config_name}.json")
    json_file = json.load(f)
    f.close()
    config = json_file[config_key]

    return config
