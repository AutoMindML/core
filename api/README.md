# AutoMind API

[![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv package manager](https://img.shields.io/badge/uv-package%20manager-6C8CFF.svg)](https://docs.astral.sh/uv/)

`automind-api` is the FastAPI service for dataset metadata, preprocessing, and model
operations. It uses the adjacent `automind` package in editable mode, SQL Server for
application data, and MindsDB for model operations.

This directory is an independent `uv` project. From here:

```powershell
uv sync --locked --dev
```

## Configure the service

Copy `.env.example` to `.env` and set the LLM profile and provider values. Configure the
SQL Server and MindsDB hosts, ports, credentials, and API binding in
`src/automind_api/configs/server.json`. The application reads that file relative to the
`api` working directory. Keep private credentials out of Git.

Startup opens connections to SQL Server and MindsDB. It is a live operation with
external side effects and requires those services, the configured ODBC driver, and the
separate MindsDB environment:

```powershell
uv run --locked --no-sync start.py
```

From the repository root, `just api-live` runs the same orchestration.

## Metadata API

Metadata routes are mounted under `/api/automl/metadata/{dataset_id}`. The generation
route is:

```text
POST /api/automl/metadata/{dataset_id}?target_column=<column>&task_type=CLASSIFICATION&force=false
```

`task_type` accepts `CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, or `REGRESSION`.
The service uses the request's `X-User-Id` session header and returns the existing
`state`, `message`, and `new_id` response semantics. The same route prefix provides
`GET /{dataset_id}` for the prompt and `GET /{dataset_id}/status` for generation and
applier status.

Metadata generation reads the dataset from SQL Server and calls the configured LLM
provider. Saving preprocessing results splits first, fits on training data, applies
SMOTE only to training data, and transforms validation data with the frozen feature
schema.

## Isolated checks

Run the API controller and service tests without starting external services:

```powershell
uv run --locked --group dev --no-sync -m pytest `
  src/automind_api/tests/test_applier_controller.py `
  src/automind_api/tests/test_llm_service.py `
  src/automind_api/tests/test_metadata_controller.py -q
uv run --locked --group dev --no-sync ruff check src
uv run --locked --group dev --no-sync pyright
```

The tests use fake LLM and persistence boundaries. Manual integration scripts under
`src/automind_api/tests/` require configured SQL Server and MindsDB and are not isolated
tests.

See the [repository guide](../README.md) for root commands and the
[core guide](../automind/README.md) for preprocessing contracts and research workflows.
