from pathlib import Path

import requests

from ...db.connection import (connect_mindsdb_server,
                              get_mindsdb_ml_engine_upload_url)

# from requests_toolbelt.multipart.encoder import MultipartEncoder

if __name__ == "__main__":
    ml_engine_name = "decision_tree_engine"
    mindsdb_server = connect_mindsdb_server()

    url = f"{get_mindsdb_ml_engine_upload_url()}/{ml_engine_name}"

    if ml_engine_name in [engine.name for engine in mindsdb_server.ml_engines.list()]:
        mindsdb_server.ml_engines.drop(ml_engine_name)

    with open(f"{Path(__file__).parent.absolute()}/iris_decision_tree.py", "rb") as f:
        with open("./requirements.txt") as f2:
            # Refer to
            # https://stackoverflow.com/questions/12385179/how-to-send-a-multipart-form-data-with-requests-in-python

            # mp_encoder = MultipartEncoder(
            #     fields={
            #         "source": "iris_decision_tree",
            #         "modules": "null",
            #         "type": "inhouse",
            #         "code": ("iris_decision_tree.py", f, "text/x-python"),
            #     }
            # )

            # headers = {"Content-Type": mp_encoder.content_type}
            # res = requests.put(url, data=mp_encoder, headers=headers)

            payload = {
                "source": "iris_decision_tree",
                "type": "inhouse",
            }

            files = {
                "code": ("iris_decision_tree.py", f, "text/x-python"),
                "modules": ("requirements.txt", f2, "text/plain"),
            }

            res = requests.put(url, data=payload, files=files)

            print(res.status_code, res.content)
