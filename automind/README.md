# AutoMind core

AutoMind core turns dataset metadata and an LLM recommendation into a fitted,
replayable preprocessing pipeline. It also contains the Data Fusion Module
(DFM), TPOT adapter, and experiment runner used to evaluate the paper's claims.

## Research pipeline

`MetaGenerator.generate_compact_llm_query()` describes column names, types, and
missing rates without including raw row values. `OpenAICompatibleProvider`
submits that prompt through one provider contract. `PreprocessingPipeline` then
validates the response, fits imputation, scaling, and encoding on training data
only, freezes the resulting feature schema, and applies sampling only to the
training partition.

`ExperimentRunner` records the protocol, dataset hash, every LLM response or
error, timing and token usage, per-run model metrics, sample standard deviations,
and preprocessing-operation Jaccard stability. A saved run can be replayed
without another LLM call.

For the paper's novice-user comparison, `PlanSelector` rejects semantically
unsafe recommendations, scores the remaining bounded candidate set with fixed
inner folds on the outer training partition, and keeps the deterministic
baseline unless a candidate exceeds the configured minimum gain. The direct
LLM-code arm can run only through an injected external sandbox executor. Its
boundary withholds holdout labels, disables network access by policy, and
rejects changed targets, row identities, or train/holdout feature schemas.
Fit receives training data only; transform runs each holdout row in a fresh
request against read-only fitted state, preventing access to holdout-distribution
statistics. `ComparisonExperiment` generates the bounded candidate set and
direct-code response through the shared provider, persists every attempt, and
then invokes the comparison runner.

Podman setup, the pinned research image, and required host isolation checks are
documented in [the direct-code sandbox guide](docs/direct-code-sandbox.md).

The supported fitted operations are:

- Missing values: mean, median, mode, zero-as-missing, negative-as-missing
- Transformation: standardization, min-max scaling, binarization
- Encoding: string index and one-hot encoding with stable unknown handling
- Training-only sampling: SMOTE and Borderline-SMOTE

Unsupported recommendations fail explicitly in strict mode.

## LLM configuration

Connection values belong in `api/.env` (copy `api/.env.example`):

```text
AUTOMIND_LLM_BASE_URL=https://arch.tailfe91b3.ts.net/v1
AUTOMIND_LLM_MODEL=Qwen3.6-35B-A3B-GGUF:MXFP4_MOE:thinking-coding
AUTOMIND_LLM_API_KEY=local
AUTOMIND_LLM_TIMEOUT=300
```

Inference parameters are versioned in `src/automind/configs/llm/`. The built-in
profiles are `local-qwen` and `openai-baseline`; select one with
`AUTOMIND_LLM_PROFILE`. The local profile selects the Qwen 3.6 thinking-coding
model and specifies JSON response mode,
temperature 0, seed 42, 8192 completion tokens, and `think: true`.

The resolution order is library defaults, JSON profile, environment/`.env`, then
explicit caller overrides. Environment values can override inference parameters
when needed; JSON-valued overrides must be JSON objects. Real `.env` files are
ignored, while `.env.example` is versioned. `LLMSettings.public_manifest()`
excludes the API key.

The same contract can use an OpenAI model by selecting the baseline profile and
changing endpoint, model, and key. `ExperimentProtocol.from_llm_settings()`
copies the resolved model and inference parameters into replayable artifacts.

## Verification

From this directory:

```powershell
uv run -m pytest ./src/automind/tests -x
$env:RUN_LIVE_LLM='1'; uv run -m pytest ./src/automind/tests/test_live_llm.py -q
$env:RUN_TPOT_SMOKE='1'; uv run -m pytest ./src/automind/tests/test_tpot_smoke.py -q
```

Live tests are skipped unless explicitly enabled. The TPOT smoke test uses a
small synthetic dataset and bounded search. Pickled TPOT artifacts must only be
loaded from trusted experiment output.

## Research protocol

The versioned Synthea pilot manifest and protocol are under
`src/automind/configs/research/`. Validate local file hashes and inspect the
bounded run matrix before making any LLM call:

```powershell
uv run -m automind.experiments validate src/automind/configs/research/synthea-covid19-pilot.protocol.json --dataset-root src/automind/data/csv/synthea_covid19_10k
uv run -m automind.experiments dry-run src/automind/configs/research/synthea-covid19-pilot.protocol.json --dataset-root src/automind/data/csv/synthea_covid19_10k
```

The current frozen pilot has C0/C2/C3, five paired repetitions, one split seed,
15 total runs, and at most 10 serial LLM calls. Dataset adapters fit the Synthea
expense threshold on training data only and use a chronological Bank Marketing
split that excludes the post-call `duration` field. Corruption operators refuse
entity keys and targets and emit a row-level ledger. Research artifacts enforce
the protocol fingerprint on resume.

The v2 novice-comparison protocol describes the primary deterministic,
direct-code, and guarded arms plus semantic-validation, inner-CV, and fallback
ablations. Its dry run reports the bounded call budget and the configured Podman
profile; readiness still requires a successful preflight on the experiment host:

```powershell
uv run -m automind.experiments dry-run src/automind/configs/research/novice-comparison-v2.protocol.json
```

After reviewing the dry run, the same protocol can execute the Synthea study:

```powershell
uv run -m automind.experiments run src/automind/configs/research/novice-comparison-v2.protocol.json --dataset-root src/automind/data/csv/synthea_covid19_10k
```

The study resolves the frozen Podman profile and refuses to run direct code when
the engine, isolation controls, or image digest fail preflight. Programmatic
tests may inject another conforming executor into `DirectCodeHarness`; generated
code is never executed directly by the host Python process.

## Adding preprocessing methods

Add the enum and Pydantic schema entry in `models/preprocessing.py`, include only
fitted implementations in the compact prompt allow-list, implement fit and
transform behavior in `pipeline/preparation.py`, and add a leakage-focused test.
The legacy interactive preview path in `LogicApplier` may support a wider set of
operations, but saved training datasets use the fitted pipeline.
