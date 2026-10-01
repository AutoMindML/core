# AutoMind core

[![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](.python-version)
[![uv](https://img.shields.io/badge/managed%20with-uv-6C8CFF?logo=uv&logoColor=white)](uv.lock)
[![Core MIT License](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

The `automind` package turns dataset metadata and an LLM recommendation into a fitted,
replayable preprocessing pipeline. It also provides data fusion, feature preparation, a
TPOT adapter, and the experiment runner.

This directory is an independent `uv` project. From here, install its locked development
environment with:

```powershell
uv sync --locked --dev
```

## Core workflow

`MetaGenerator` describes columns, types, and missingness without sending raw row values
in the compact prompt. `OpenAICompatibleProvider` supplies one provider contract.
`PreprocessingPipeline` validates the response, fits imputation, scaling, and encoding
on training data, freezes the feature schema, and applies sampling only to the training
partition.

The supported fitted operations are:

- Missing values: mean, median, mode, zero-as-missing, negative-as-missing
- Transformation: standardization, min-max scaling, and binarization
- Encoding: string index and one-hot encoding with stable unknown handling
- Training-only sampling: SMOTE and Borderline-SMOTE

Unsupported recommendations fail explicitly in strict mode. The direct-code condition
uses an injected external sandbox executor. Its policy protects holdout labels and row
identities, disables network access, and checks the target and feature schema before
executing generated code.

## LLM configuration

Copy [the API example environment file](../api/.env.example) to a private `api/.env` and
set the provider values for your environment. When invoking an experiment from
`automind/`, point `AUTOMIND_ENV_FILE` at that file's absolute path, for example
`$env:AUTOMIND_ENV_FILE = (Resolve-Path ../api/.env).Path` in PowerShell. Select a
versioned profile with `AUTOMIND_LLM_PROFILE`; the built-in profiles are `local-qwen`
and `openai-baseline`. The resolution order is library defaults, JSON profile,
environment values, then explicit caller overrides.

Inference profiles live in `src/automind/configs/llm/`. Do not commit API keys or
private endpoints. `ExperimentProtocol.from_llm_settings()` records the resolved model
and inference parameters in replayable artifacts.

## Research protocol

The maintained Synthea direct-code preset is
`src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json`. Its
condition, seeds, model profile, and execution budget are defined in the JSON. The
dataset manifest and CSV fixture are separate local resources under
`src/automind/configs/research/` and `src/automind/data/csv/synthea_covid19_10k/`.

Inspect a protocol without making an LLM request:

```powershell
uv run --locked --group dev --no-sync automind-experiment validate `
  src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json `
  --dataset-root src/automind/data/csv/synthea_covid19_10k

uv run --locked --group dev --no-sync automind-experiment dry-run `
  src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json `
  --dataset-root src/automind/data/csv/synthea_covid19_10k
```

These commands read local manifests and, for a direct-code protocol, perform Podman
preflight checks. They do not call the LLM or train a model. A live run calls the
configured provider, executes generated code in Podman, and writes under the protocol
output root:

```powershell
uv run --locked --group dev --no-sync automind-experiment run `
  src/automind/configs/research/synthea-covid19-direct-code-v2.protocol.json `
  --dataset-root src/automind/data/csv/synthea_covid19_10k
```

Use the [direct-code sandbox guide](docs/direct-code-sandbox.md) before a live run. To
continue a matching run, replace `run` with `resume` in the command above. The
repository `justfile` also exposes summarize and replay recipes. Protocol JSON identity
fields participate in the fingerprint, so changing them requires a new output root;
renaming the file alone preserves the existing results.

## Verification

The isolated core suite uses the checked-in test paths and local fixtures:

```powershell
uv run --locked --group dev --no-sync -m pytest ./src/automind/tests `
  -m "not live_llm and not tpot_smoke and not podman_sandbox" -x
uv run --locked --group dev --no-sync ruff check src
uv run --locked --group dev --no-sync pyright
```

The isolated command excludes live LLM, TPOT smoke, and Podman tests. The Podman
suite is available through `just experiment-sandbox-check` from the repository
root; live LLM and TPOT smoke tests require their documented environment flags.

## Extending preprocessing

When adding a preprocessing method, update the enum and schema in
`src/automind/models/preprocessing.py`, the prompt examples and allow-list, the
registered implementation, the logic applier, and its focused tests. Keep fit and
transform behavior aligned with the training-only data contract.
