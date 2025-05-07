import unittest
from pathlib import Path
from time import sleep

import requests


class TestAPI(unittest.TestCase):
    api_url = "http://127.0.0.1:8000"
    data_dir = f"{Path(__file__).parent.parent.absolute()}/src/data"

    # in mssql
    member_id = 2
    engine_id = 86

    # in mindsdb
    # engine name: 54B598504454FCBF5503ABF3636D9986

    connection_args = {
        # "host": "host.docker.internal",
        "host": "127.0.0.1",
        "port": 1433,
        "user": "automl",
        "password": ".automl.",
        "database": "AutoML",
    }

    def test_01_1_add_data_source_file(self):
        with open(f"{self.data_dir}/iris.csv", "rb") as f:
            res = requests.post(
                self.api_url + "/api/data/file",
                headers={"Content-Type": None},
                data={
                    "mid": self.member_id,
                    "name": "test: data source file",
                    "des": "data source file",
                },
                files={"file": ("iris.csv", f, "text/csv")},
            )

            self.assertEqual(res.status_code, 200)

            res_data = res.json()
            self.assertNotEqual(res_data["new_id"], None)

            TestAPI.data_file_id = res_data["new_id"]

    def test_01_2_add_data_source_database(self):
        res = requests.post(
            self.api_url + "/api/data/database",
            json={
                "mid": self.member_id,
                "name": "test: data source database",
                "des": "data source database",
                "engine": "mssql",
                "connection_args": self.connection_args,
            },
        )

        self.assertEqual(res.status_code, 200)

        res_data = res.json()
        self.assertNotEqual(res_data["new_id"], None)

        TestAPI.data_database_id = res_data["new_id"]

    def test_01_3_get_data_source(self):
        res = requests.get(
            self.api_url + "/api/data",
            params={"mid": self.member_id, "oid": TestAPI.data_file_id},
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["accept-ranges"], "bytes")

    def test_02_add_project(self):
        res = requests.post(
            self.api_url + "/api/project",
            json={"mid": self.member_id, "name": "test: project", "des": "project"},
        )

        self.assertEqual(res.status_code, 200)

        res_data = res.json()
        self.assertNotEqual(res_data["new_id"], None)

        TestAPI.project_id = res_data["new_id"]

    def test_03_add_model(self):
        res = requests.post(
            self.api_url + "/api/model",
            json={
                "mid": self.member_id,
                "cid": TestAPI.project_id,
                "data_oid": TestAPI.data_file_id,
                "engine_oid": self.engine_id,
                "name": "test: model",
                "des": "model",
                "predict": "Species",
                "tag": "test:model",
                "select_data_query": "",
                "training_options": "",
            },
        )

        self.assertEqual(res.status_code, 200)

        res_data = res.json()
        self.assertNotEqual(res_data["new_id"], None)

        TestAPI.model_id = res_data["new_id"]

    def test_04_1_add_app(self):
        res = requests.post(
            self.api_url + "/api/app/prediction",
            json={
                "mid": self.member_id,
                "project_id": TestAPI.project_id,
                "model_id": TestAPI.model_id,
                "name": "test: app",
                "des": "app prediction",
            },
        )

        self.assertEqual(res.status_code, 200)

        res_data = res.json()
        self.assertNotEqual(res_data["new_id"], None)

        TestAPI.app_id = res_data["new_id"]
        TestAPI.api_key = res_data["api_key"]

    def test_04_2_use_app(self):
        sleep(10)

        input = [
            {
                "SepalLengthCm": 6.3,
                "SepalWidthCm": 3.3,
                "PetalLengthCm": 4.7,
                "PetalWidthCm": 1.6,
            },
            {
                "SepalLengthCm": 6.5,
                "SepalWidthCm": 3,
                "PetalLengthCm": 5.8,
                "PetalWidthCm": 2.2,
            },
            {
                "SepalLengthCm": 4.7,
                "SepalWidthCm": 3.2,
                "PetalLengthCm": 1.6,
                "PetalWidthCm": 0.2,
            },
        ]

        res = requests.post(
            self.api_url + "/api/service/app",
            headers={"user-agent": "Mozilla/5.0", "api-key": TestAPI.api_key},
            json=input,
        )

        self.assertEqual(res.status_code, 200)

    def test_05_update(self):
        res = requests.patch(
            self.api_url + "/api/object/update",
            json={
                "mid": self.member_id,
                "oid": TestAPI.data_file_id,
                "name": "update data source",
                "des": "update data source",
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

        res = requests.patch(
            self.api_url + "/api/class",
            json={
                "mid": self.member_id,
                "cid": TestAPI.project_id,
                "name": "update project",
                "des": "update project",
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

    def test_06_delete(self):
        res = requests.patch(
            self.api_url + "/api/object/hide",
            json={
                "mid": self.member_id,
                "oid": TestAPI.data_file_id,
                "action": "hide",
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 1)

        res = requests.delete(
            self.api_url + "/api/data",
            json={
                "mid": self.member_id,
                "oid": TestAPI.data_file_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 1)

        res = requests.delete(
            self.api_url + "/api/project",
            json={
                "mid": self.member_id,
                "cid": TestAPI.project_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 1)

        res = requests.delete(
            self.api_url + "/api/model",
            json={
                "mid": self.member_id,
                "project_id": TestAPI.project_id,
                "model_id": TestAPI.model_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 1)

        res = requests.delete(
            self.api_url + "/api/app/prediction",
            json={
                "mid": self.member_id,
                "app_id": TestAPI.app_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

        res = requests.delete(
            self.api_url + "/api/model",
            json={
                "mid": self.member_id,
                "project_id": TestAPI.project_id,
                "model_id": TestAPI.model_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

        res = requests.delete(
            self.api_url + "/api/project",
            json={
                "mid": self.member_id,
                "cid": TestAPI.project_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

        res = requests.patch(
            self.api_url + "/api/object/hide",
            json={
                "mid": self.member_id,
                "oid": TestAPI.data_file_id,
                "action": "hide",
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

        res = requests.patch(
            self.api_url + "/api/object/hide",
            json={
                "mid": self.member_id,
                "oid": TestAPI.data_file_id,
                "action": "restore",
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

        res = requests.delete(
            self.api_url + "/api/data",
            json={
                "mid": self.member_id,
                "oid": TestAPI.data_file_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)

        res = requests.delete(
            self.api_url + "/api/data",
            json={
                "mid": self.member_id,
                "oid": TestAPI.data_database_id,
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
