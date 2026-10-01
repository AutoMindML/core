# AutoMind Core

[![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv package manager](https://img.shields.io/badge/uv-package%20manager-6C8CFF.svg)](https://docs.astral.sh/uv/)

AutoMind Core is a two-package workspace for LLM-assisted preprocessing and model
workflows. `automind` is the reusable Python package and experiment runner;
`automind-api` is the FastAPI service that connects preprocessing to SQL Server and
MindsDB.

The packages are separate `uv` projects. Each has its own `pyproject.toml`, `uv.lock`,
and `.venv`. The API uses the adjacent core package as an editable dependency. There is
no Python project at the repository root.

## Choose a workflow

- [Core package and research workflows](automind/README.md)
- [API service and metadata endpoint](api/README.md)
- [Direct-code sandbox procedure](automind/docs/direct-code-sandbox.md)

## Prerequisites

Install Python 3.10, [uv](https://docs.astral.sh/uv/), and
[`just`](https://github.com/casey/just) to use the commands below. Research direct code
additionally requires Podman and the local Synthea fixture described in the core guide.

## Set up and check the workspace

From this directory, synchronize both locked development environments:

```powershell
just sync
just --list
```

The command list confirms that `just` can read this workspace. Sync downloads
dependencies and writes `automind/.venv` and `api/.venv`. Without `just`, run
`uv sync --locked --dev` separately from each package directory.

Run the isolated checks and builds with:

```powershell
just test
just lint
just typecheck
just build
```

Core research tests read the untracked Synthea CSV files under
`automind/src/automind/data/csv/synthea_covid19_10k/`. The default recipes exclude live
LLM, TPOT smoke, and Podman integration tests. Builds write archives under the ignored
root `dist/` directory.

## Run the API locally

Copy `api/.env.example` to `api/.env`, set the LLM values, and configure the SQL Server
and MindsDB connections in `api/src/automind_api/configs/server.json`. The service
startup connects to those systems, so starting it is a live operation with external side
effects:

```powershell
just api-live
```

Use the [API guide](api/README.md) for the mounted routes and request examples.

## Validate or run the research protocol

The maintained research preset is
`automind/src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json`.
It runs the direct-code condition using the Synthea dataset manifest. From the
repository root:

```powershell
just experiment-validate src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json src/automind/data/csv/synthea_covid19_10k
just experiment-dry-run src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json src/automind/data/csv/synthea_covid19_10k
```

Validation and dry run inspect local files and sandbox readiness. Running or resuming
the protocol calls the configured LLM, executes generated code in the Podman sandbox,
and writes research artifacts; review the
[sandbox guide](automind/docs/direct-code-sandbox.md) before using `experiment-run-live`
or `experiment-resume-live`.

## Development

Package-specific commands, data contracts, protocol details, and contribution guidance
are in the [core guide](automind/README.md) and [API guide](api/README.md).
