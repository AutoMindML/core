set shell := ["powershell.exe", "-NoProfile", "-Command"]

default: help

help:
    @Write-Output "Core recipes: sync, build, test, lint, typecheck, api-live, experiment-*"
    @Write-Output "experiment-sandbox-check runs the Podman isolation suite."
    @Write-Output "experiment-dry-run performs protocol and Podman preflight checks."

sync: api-sync automind-sync

api-sync:
    Set-Location api; uv sync --locked --dev

automind-sync:
    Set-Location automind; uv sync --locked --dev

build: api-build automind-build

api-build: api-sync
    Set-Location api; uv build --out-dir ../dist/api

automind-build: automind-sync
    Set-Location automind; uv build --out-dir ../dist/automind

test: automind-test api-test

automind-test: automind-sync
    Set-Location automind; uv run --locked --group dev -m pytest ./src/automind/tests -m "not live_llm and not tpot_smoke and not podman_sandbox" -x

api-test: api-sync
    Set-Location api; uv run --locked --group dev -m pytest src/automind_api/tests/test_applier_controller.py src/automind_api/tests/test_llm_service.py src/automind_api/tests/test_metadata_controller.py -x

lint: automind-lint api-lint

automind-lint: automind-sync
    Set-Location automind; uv run --locked --group dev ruff check src

api-lint: api-sync
    Set-Location api; uv run --locked --group dev ruff check src

typecheck: automind-typecheck api-typecheck

automind-typecheck: automind-sync
    Set-Location automind; uv run --locked --group dev pyright

api-typecheck: api-sync
    Set-Location api; uv run --locked --group dev pyright

api-live:
    Set-Location api; uv run --locked --no-sync start.py

experiment-help:
    Set-Location automind; uv run --locked --group dev --no-sync automind-experiment --help

experiment-validate protocol dataset_root:
    Set-Location automind; uv run --locked --group dev --no-sync automind-experiment validate {{ quote(protocol) }} --dataset-root {{ quote(dataset_root) }}

experiment-sandbox-check:
    Set-Location automind; uv run --locked --group dev --no-sync -m pytest src/automind/tests/test_podman_sandbox.py -v

experiment-dry-run protocol dataset_root="":
    Set-Location automind; uv run --locked --group dev --no-sync automind-experiment dry-run {{ quote(protocol) }} {{ if dataset_root != "" { "--dataset-root " + quote(dataset_root) } else { "" } }}

experiment-run-live protocol dataset_root:
    Set-Location automind; uv run --locked --group dev --no-sync automind-experiment run {{ quote(protocol) }} --dataset-root {{ quote(dataset_root) }}; if ($LASTEXITCODE -eq 130) { Write-Output 'Experiment interrupted; use experiment-resume-live to continue.'; exit 0 }; exit $LASTEXITCODE

experiment-resume-live protocol dataset_root:
    Set-Location automind; uv run --locked --group dev --no-sync automind-experiment resume {{ quote(protocol) }} --dataset-root {{ quote(dataset_root) }}; if ($LASTEXITCODE -eq 130) { Write-Output 'Experiment interrupted; use experiment-resume-live to continue.'; exit 0 }; exit $LASTEXITCODE

experiment-summarize output_root:
    Set-Location automind; uv run --locked --group dev --no-sync automind-experiment summarize {{ quote(output_root) }}

experiment-replay completion output_root metadata="":
    Set-Location automind; uv run --locked --group dev --no-sync automind-experiment replay-v2 {{ quote(completion) }} {{ quote(output_root) }} {{ if metadata != "" { "--metadata " + quote(metadata) } else { "" } }}
