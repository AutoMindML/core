# AGENTS Instructions

## Repository boundaries and ownership

- These instructions apply to the entire AutoMind Core repository. Check `git status --short` before
  editing and preserve existing changes. Investigate before resetting, cleaning, or overwriting outputs.
- `automind/` and `api/` are separate Python 3.10 projects, each with its own `pyproject.toml`,
  `uv.lock`, and `.venv`. The root `justfile` wraps their commands; it is not a shared Python project.
  The API uses the local `../automind/` package as an editable dependency. Put reusable behavior in the
  core package.
- In `automind/src/automind/`, `data_utils/` and `models/` own data fusion, column parsing, legacy
  data-cleaning and feature-engineering methods, and shared schemas. `pipeline/` owns fitted
  preprocessing, semantic validation, and candidate selection. `service/` owns shared LLM configuration,
  providers, and response contracts.
- `automind/src/automind/experiments/` owns dataset adapters, protocols, execution, evaluation,
  sandboxing, progress, and artifacts. `configs/llm/` and `configs/research/` contain versioned
  configuration. TPOT wrappers live in `engine/`; evaluation logic also lives in `evaluation/`.
- In `api/src/automind_api/app/`, `controllers/` defines HTTP routes, `services/` orchestrates workflows,
  `repositories/` handles data access, and `models/` defines request and database mappings. `db/` and
  `configs/` own connections and settings; `api/create-db/` owns SQL schemas, views, and procedures. The
  core package must not depend on the API.
- Use `api/src/automind_api/main.py` as the authority for mounted routes, and
  `api/src/automind_api/app/models/view_sp.py` plus the SQL sources for view and procedure names. Check
  both `api/start.py` and `api/__main__.py` when changing API or MindsDB startup. Their `../../mindsdb/`
  dependency is outside this repository.

## Data, LLM, and experiment contracts

- When changing a legacy data-cleaning or feature-engineering method, check the enums and schemas in
  `models/preprocessing.py`, prompts and examples in `data_utils/template.py`, the `@register_method`
  implementation in `data_utils/preprocessing.py`, dispatch in `data_utils/logic_applier.py`, relevant
  tests, and the core README. `EnumByName` and the method registry use enum **names** as keys; preserve
  names and avoid collisions.
- Legacy LLM recommendations use `<json>...</json>`. A format change must update the parser, schema,
  prompt, and fixture together. Preserve `LogicApplier`'s input DataFrame copy, separation of original
  and processed data, and index, target, and feature alignment.
- Fit preprocessing only on training data and sample only the training partition. Validation and holdout
  data must use frozen fitted state and feature schema. Changes to `pipeline/` or experiment comparison
  must test leakage protection, column alignment, and explicit failure for unsupported recommendations.
- Process ordinary LLM recommendations through schemas and registered methods; do not directly `eval` or
  `exec` response text. Research `direct_code` may run only in the controlled sandbox, never in the host
  Python process. Preserve row identity, target protection, single-row holdout handling, read-only
  transform state, and network isolation.
- Name protocol files `<dataset>-<experiment>-v<schema>.protocol.json`. A filename may change without
  changing protocol identity, but JSON fields such as `name` and `output_root` affect the fingerprint.
  Changes to those fields require a new run identity and output location; `resume` must reject mismatched
  artifacts. Retain interruption markers, attempt journals, and failure artifacts for diagnosis. Do not
  edit results to make a run appear successful.

## API and configuration contracts

- When changing a stored procedure, check its Python parameter mapping: `exec_mutation_sp()` names must
  match SQL parameters. When changing a view, update its TypedDict fields; `get_view_by_id()` builds
  queries from schema annotations.
- Preserve the meanings of `state`, `message`, and `new_id` in HTTP responses, and check controllers,
  models, and callers when changing the contract. Bind SQL values as parameters; dynamic identifiers must
  come from controlled mappings or validation.
- `HeaderSessionMiddleware` derives request state from `X-User-Id`; `env == "test"` uses a fixed
  identity. Do not switch production settings to test mode to bypass validation, and preserve deployed
  API key checks.
- API settings resolve `src/automind_api/configs/` from the working directory. Some legacy Azure modules
  create clients at import time. Shared LLM settings resolve profiles, environment variables, and
  `AUTOMIND_ENV_FILE` in `service/config.py`. Core `utils/config.py` also uses the working directory;
  inspect callers before changing path resolution.
- `.env`, `private.*`, server and MindsDB settings, and SQL or batch scripts may contain secrets or
  machine-specific data. Read only the necessary structure; never print values in tool output or copy
  them into documentation, tests, or commits. Protect accidentally tracked secrets as well.

## Validation and side effects

- Inspect a `just` recipe's working directory and effects before running it. `just sync`, `just test`,
  `just lint`, and `just typecheck` may sync both environments. Documentation changes do not require
  installation, service startup, or model training.
- Run isolated core tests from `automind/` with
  `uv run --locked --group dev --no-sync -m pytest ./src/automind/tests -m "not live_llm and not tpot_smoke and not podman_sandbox" -x`.
  This requires an existing environment and local CSV fixtures under
  `src/automind/data/csv/synthea_covid19_10k/`; the data is not tracked and empty substitutes are
  invalid. Some tests write `llm_query.txt`.
- The root `just api-test` recipe selects isolated API tests. `api/src/automind_api/tests/` also contains
  manual integration scripts, so do not treat the whole directory as an offline unit-test suite. Isolate
  the database, MindsDB, LLM, lifespan, and import-time client initialization in new API tests.
- For cross-module data-flow or schema changes, run focused checks and the isolated core suite. For
  sandbox-contract changes, also run `just experiment-sandbox-check`. Documentation and protocol
  organization can be checked with path-specific validation, links, and `git diff --check`. A v2 dry run
  performs Podman preflight and is not purely static. Report missing prerequisites and checks not run; do
  not describe static checks as passed tests.
- `just api-live` connects to SQL Server and starts external MindsDB. `just experiment-run-live` and
  `just experiment-resume-live` make real LLM requests, train models, and write artifacts. Do not
  initialize databases, execute DDL, modify or delete remote data, upload engines, or make real Azure
  requests without authorization for those actions. Honor authorization already given in the current
  conversation.
- Batch and SQL scripts in `api/create-db/` may change databases and accounts; some SQL enables
  `TRUSTWORTHY`. `api/db-setup.bat` still references a missing sqlpredictor path. Experiments may
  overwrite CSVs, reports, and models; use mocks and temporary directories in tests, and do not load
  untrusted pickle files.
- Do not manually edit `.venv/`, `*.egg-info/`, caches, generated typings, or binary DLLs, or regenerate
  research outputs incidentally. Update the relevant README when changing a public workflow. Sandbox
  operations belong in `automind/docs/direct-code-sandbox.md`; agent editing rules belong here. Before
  committing, verify that only task-related files are included and that no secrets, datasets, or model
  artifacts were added.
