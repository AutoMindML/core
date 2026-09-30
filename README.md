# AutoMind Core

AutoMind Core contains the reusable `automind` Python package and the FastAPI
`automind-api` service. The API coordinates preprocessing, SQL Server data, and
MindsDB model operations. The core package also contains the research experiment
runner. See the [core guide](automind/README.md) and [API guide](api/README.md)
for their respective workflows.

The two directories are **separate uv projects** with their own `pyproject.toml`,
`uv.lock`, and `.venv`. The API installs `../automind` as an editable local
dependency. There is no Python project at this repository root.

## Get started

On Windows, install Python 3.10, [uv](https://docs.astral.sh/uv/), and
[just](https://github.com/casey/just). From this repository root, synchronize
both locked development environments and list the available commands:

```powershell
just sync
just --list
```

`just sync` writes to `api/.venv` and `automind/.venv` and may download packages.
The `justfile` runs each command in its owning project directory. To work without
just, run `uv sync --locked --dev` separately from `automind/` and `api/`, or run
the repository's `setup.ps1` from PowerShell.

Run the isolated checks and build both package distributions:

```powershell
just test
just lint
just typecheck
just build
```

The core tests need the untracked Synthea CSV fixtures under
`automind/src/automind/data/csv/synthea_covid19_10k/`. The default test recipes
exclude live LLM, TPOT smoke, and Podman sandbox tests. API tests use fake
external services; manual integration scripts are excluded. The build recipe
writes package archives under the ignored `dist/` directory.

## Run the service

Copy `api/.env.example` to `api/.env` and configure the local SQL Server and
MindsDB connections before starting the service. From the repository root:

```powershell
just api-live
```

This calls `api/start.py`, which starts both the API and the external MindsDB
environment. Startup can connect to SQL Server, start MindsDB jobs, and apply
its own migrations. See the [API guide](api/README.md) for service details.

## Run experiments

The `justfile` exposes experiment validation, dry run, execution, resume,
summary, and replay recipes. Protocol and dataset paths passed to these recipes
are relative to `automind/` unless absolute paths are supplied. For example,
with the local Synthea files present:

```powershell
just experiment-validate src/automind/configs/research/synthea-covid19-pilot.protocol.json src/automind/data/csv/synthea_covid19_10k
just experiment-dry-run src/automind/configs/research/synthea-covid19-pilot.protocol.json src/automind/data/csv/synthea_covid19_10k
```

For the v2 comparison protocol, run the Podman isolation suite, then check
protocol and dataset readiness with a dry run:

```powershell
just experiment-sandbox-check
just experiment-dry-run src/automind/configs/research/novice-comparison-v2-confirmatory.protocol.json src/automind/data/csv/synthea_covid19_10k
```

The dry run should report `direct_code_ready: true` and
`sandbox_readiness: ready`. The `experiment-run-live` and
`experiment-resume-live` recipes can call an LLM, train models, and write
research output. Review the protocol and use explicit arguments when running
them. Start the confirmatory v2 protocol with `experiment-run-live`; use
`experiment-resume-live` with the same arguments after an interrupted run:

```powershell
just experiment-run-live src/automind/configs/research/novice-comparison-v2-confirmatory.protocol.json src/automind/data/csv/synthea_covid19_10k
just experiment-resume-live src/automind/configs/research/novice-comparison-v2-confirmatory.protocol.json src/automind/data/csv/synthea_covid19_10k
```

The confirmatory v2 protocol uses a separate output root because the earlier
v2 root contains results with a different run identity. Protocols and the
research procedure are documented in the
[core guide](automind/README.md).
