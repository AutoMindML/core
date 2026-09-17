# AutoMind API

The FastAPI service coordinates datasets in SQL Server, core preprocessing, and
MindsDB model operations. Metadata generation uses the shared
`OpenAICompatibleProvider`; it does not create a remote client during import.

Copy `.env.example` to `.env` and set the connection values. The selected JSON
profile supplies versioned inference parameters; environment variables may
override them. Configuration precedence is defaults, profile, environment, then
explicit caller overrides. Real dotenv files are excluded from Git.

`POST /metadata/{dataset_id}` accepts `target_column`, `force`, and `task_type`.
The task type is `CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, or `REGRESSION`.
The stored LLM envelope includes schema version, content, model, finish reason,
token usage, and elapsed time. Existing Azure/OpenAI-style stored responses are
still readable.

Saving preprocessing results splits the source data first, fits preprocessing on
the training partition, applies SMOTE only to training data, and transforms the
validation partition with the frozen feature schema.

Run isolated API tests from this directory:

```powershell
uv run -m pytest src/automind_api/tests/test_llm_service.py src/automind_api/tests/test_metadata_controller.py -q
```

These tests use fake LLM and persistence boundaries and do not start the API,
connect to SQL Server, or call MindsDB.
