# MlSecOps_project

Predictive Maintenance / MLSecOps project by Mate, Anna, and Beni.

## Project structure

```text
MlSecOps_project/
├── app/                         # Streamlit monitoring dashboard
├── data/                        # Dataset files, not Python modules
│   ├── raw/                     # Original CSV, managed by DVC
│   ├── raw.dvc                  # DVC metadata for the raw dataset
│   └── processed/               # Processed and versioned datasets
│       └── training/            # DVC-managed training datasets
├── docs/                        # API contract, workflow, and component handoffs
├── infra/                       # Infrastructure configuration
├── scripts/                     # Training, export, monitoring, and verification
├── src/
│   └── predictive_maintenance/
│       ├── api/                 # FastAPI routes, schemas, and API services
│       ├── data/                # Loading, preprocessing, splitting, and provenance
│       ├── models/              # Training, evaluation, and MLflow Registry
│       └── monitoring/          # Prediction storage, export, and drift monitoring
├── tests/
│   ├── unit/                    # Component and isolated API/service tests
│   └── integration/             # Dataset and MLflow integration tests
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── pytest.ini
└── README.md
```

The root-level `data/` directory holds datasets. References such as
`data/loader.py` in component handoffs are relative to
`src/predictive_maintenance/`, not the repository root.

GitHub stores source code, configuration, documentation, tests, and DVC
metadata. Dataset contents are stored in the shared DagsHub DVC remote.

- Storage project: https://dagshub.com/bsumicz/MlSecOps_project
- DVC remote name: `dagshub` (default)
- Raw dataset metadata: `data/raw.dvc`
- Raw dataset path: `data/raw/predictive_maintenance_5000.csv`
- Training dataset metadata: `data/processed/training/train_v1.csv.dvc`
  and `data/processed/training/train_v2.csv.dvc`

Do not commit dataset contents or credentials to Git.
The processed submission export is versioned separately with
`data/processed/submission.dvc`; see the [dataset release and export guide](docs/dataset_release.md).

## Local setup (PowerShell)

Run commands from the repository root with the project virtual environment
activated. The CI environment and Docker image use Python 3.11.

Install the project dependencies and development tools:

```powershell
python -m pip install -r requirements.txt
python -m pip install "dvc[s3]==3.67.1" "pre-commit==4.7.0"
```

Each team member should create a DagsHub account and obtain access to the
storage project. Use your own DagsHub access token.

Enter the token in the Password field of the credential dialog:

```powershell
$dagshubCredential = Get-Credential -UserName "token" -Message "Enter your DagsHub token in the Password field"
```

After confirming the dialog, configure local authentication:

```powershell
dvc remote modify --local dagshub access_key_id ($dagshubCredential.GetNetworkCredential().Password)
dvc remote modify --local dagshub secret_access_key ($dagshubCredential.GetNetworkCredential().Password)
Remove-Variable dagshubCredential
```

Credentials are stored in the Git-ignored `.dvc/config.local` file.
Never commit this file or share your token.

Download the datasets referenced by the current Git checkout:

```powershell
dvc pull -r dagshub
dvc status
```

### Updating tracked data

After intentionally changing the raw dataset:

```powershell
dvc add data/raw
dvc push -r dagshub
git diff -- data/raw.dvc
```

Review and commit the updated DVC metadata. Upload the corresponding data
objects before pushing the Git commit.

### CI data access

GitHub Actions installs DVC with S3 support and downloads the raw dataset
before running tests. Authentication uses the `DAGSHUB_TOKEN` repository secret.

Missing credentials, failed downloads, or a missing expected CSV fail the
workflow. Dataset-dependent integration tests are not conditionally skipped.

## Pre-commit checks

Install the Git hook separately in each local clone:

```powershell
python -m pre_commit install
```

Run all checks manually:

```powershell
python -m pre_commit run --all-files
```

Checks cover trailing whitespace, file endings, YAML syntax, merge conflict
markers, large added files, and Flake8 using the existing `.flake8` rules.

If a hook modifies files, review the changes, stage the corrected files,
and retry the commit. Full tests and coverage remain separate checks.

## Running the API locally

Run all commands from the repository root.
The commands below use the Python environment selected by `python`.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Start MLflow in the first terminal

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --workers 1
```

Keep this terminal running. The MLflow UI is available at
http://127.0.0.1:5000.

Run metadata is stored in `mlflow.db` and artifacts in `mlartifacts/`.
These generated files are Git-ignored. Keep the same database and artifacts
between restarts to retain access to logged models.

### Start the API in a second terminal

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m uvicorn predictive_maintenance.api.main:app --app-dir src --reload
```

The `--app-dir src` option makes the application package importable.
The `--reload` option reloads the development server when source files change.
The repository root is also required because training logs `src/` with the model.

- API base URL: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI specification: http://127.0.0.1:8000/openapi.json

The server can start before a champion exists. `/health` checks API
responsiveness; `/predict` requires a reachable Registry champion.

### Train and choose a prediction model

#### Server-side training settings

The API reads these environment variables when a service operation runs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRAINING_REPOSITORY_ROOT` | Application repository root | Repository containing the training datasets and Git/DVC metadata |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Shared MLflow backend for API training and prediction |

Normal local development does not require `TRAINING_REPOSITORY_ROOT`.
To use a separate training checkout, set its absolute path in the API terminal
before starting the server:

```powershell
$env:TRAINING_REPOSITORY_ROOT = "C:/path/to/MlSecOps_project"
```

Both training versions require
`data/raw/predictive_maintenance_5000.csv` beneath that root.
Version `"v2"` additionally requires
`data/processed/training/train_v2.csv`.

The root must be absolute, and explicitly empty or whitespace-only settings
are rejected. The HTTP request cannot override either server setting.
Changing the training root does not move the prediction-event database.

The API calls `train_from_dvc(...)` with these server-controlled settings.
Before training, restore the required data and verify that the training
checkout is clean:

```powershell
python -m dvc pull data/raw.dvc data/processed/training/train_v2.csv.dvc -r dagshub
git status --short
```

Install DVC with S3 support and configure personal credentials as described
above. Commit or stash changes before training; a dirty training checkout
is rejected by provenance validation.

The API still runs from its application repository root because model logging
uses its relative `src/` code path. A separate training checkout supplies the
datasets and their Git/DVC provenance; it does not change the running API code.

#### Training request

In Swagger UI, call `POST /train` with:

```json
{
  "dataset_id": "default",
  "dataset_version": "v1",
  "model_type": "gradient_boosting",
  "random_state": 42
}
```

`dataset_version` accepts `"v1"` or `"v2"` and defaults to `"v1"`.

- `"v1"` uses the initial training partition.
- `"v2"` uses the verified expanded labeled training dataset.

Both versions preserve the original validation and final-test partitions.

To request `"v2"` with the other default settings:

```json
{
  "dataset_version": "v2"
}
```

An empty object `{}` uses all defaults, including `"v1"`.
Unsupported versions and explicit null values return HTTP `422`.

`xgboost` is also supported. Training verifies the Git/DVC data, logs the
model, registers a candidate version and runs audited champion selection.

The response includes the run, logged-model URI, six validation metrics,
candidate version, selected champion version, `promoted`, and `selection_reason`.
The response does not include a `dataset_version` field.

Dataset versions (`"v1"` and `"v2"`) are separate from numeric MLflow
Registry model versions.

Inspect the run in `predictive-maintenance-baselines` and the registered model
`predictive-maintenance-model` in MLflow. The `champion` alias identifies the
currently selected version.

Promotion requires recall of at least 0.70 and, when a champion exists,
a strictly higher F1 score on the validation partition.
An equal or worse candidate stays registered without replacing the champion.

A completed training operation does not guarantee promotion, including when
using `"v2"`. If the first candidate is rejected, `champion_version` is null
and prediction remains unavailable.

On successful promotion, the next prediction resolves the new version
automatically; no API restart or `MODEL_URI` update is required.

Environment variables apply to the terminal and its child processes.
Set the server settings before startup and restart the API after changing them.

### Make a real prediction

In Swagger UI, call `POST /predict` with:

```json
{
  "type": "L",
  "air_temperature_k": 300.8,
  "process_temperature_k": 310.3,
  "rotational_speed_rpm": 1538,
  "torque_nm": 36.1,
  "tool_wear_min": 198
}
```

The fitted pipeline performs preprocessing and calculates the failure
probability. The response uses `mode: "real"`.

Prediction `1` means the probability is at least `0.5`;
otherwise the prediction is `0`.

`model_version` contains the concrete Registry version, for example `"2"`.

Before each prediction, the API resolves
`predictive-maintenance-model@champion`
and loads `models:/predictive-maintenance-model/<version>`.

The latest concrete URI/backend pair is cached per process.
Unchanged versions reuse the model; promotion or rollback selects the
corresponding version on the next request.

`MODEL_URI` is no longer used by the API.
The API does not fall back to a stale cached model if alias lookup fails.

A missing champion returns `503` with
`No champion model is available.`;
backend failures return a controlled `503`.

A request already using a resolved version completes with that same version
even if the alias changes during processing. Only use trusted model artifacts.

## Running with Docker Compose

Docker Compose starts the API, MLflow and the Streamlit monitoring dashboard.
A preparation service restores a committed training checkout and its DVC data.

Configure `.env` using `.env.example`, then run:

```powershell
docker compose config --quiet
docker compose build api
docker compose up -d --no-build
docker compose ps -a
```

All services use the same image, so building `api` once prepares the image
for the entire stack.

- Swagger: http://127.0.0.1:8000/docs
- MLflow: http://127.0.0.1:5000
- Dashboard: http://127.0.0.1:8501

See the [Docker instructions](docs/docker.md) for startup, persistent storage,
verification and troubleshooting.

## Persistent prediction events

Validated `/predict` requests produce a committed SQLite event before a
successful response is returned.

The default storage path is `artifacts/predictions.db`, which is Git-ignored.
The directory and table are created automatically on first use.

Override the location in the API terminal before starting the server when needed:

```powershell
$env:PREDICTION_LOG_DB = "C:/path/to/predictions.db"
```

The `X-Request-ID` response header identifies the stored event.

Events include a UTC timestamp, validated input features, Registry name
and alias, concrete model URI for successful predictions, threshold,
status, and the prediction result or public error message.

Invalid `422` requests are rejected before this logging step.
A storage failure returns `503` with
`The prediction event could not be stored.`
rather than reporting an unaudited prediction as successful.

Historical events retain their original values.
Reusing the same database file preserves committed events across API restarts.

Docker Compose mounts prediction storage on the persistent `prediction_data`
volume. No pruning or retention policy is implemented.

Prediction artifacts are uploaded by the separate export command below;
a prediction request does not upload a batch.

## Export prediction snapshots to MLflow

Keep the MLflow server running. In a separate PowerShell terminal at the
repository root, configure the same tracking backend used by the API:

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python scripts/export_predictions.py
```

The default source is `artifacts/predictions.db`, or `PREDICTION_LOG_DB` when
configured. `--database` and `--tracking-uri` explicitly override those choices.

If no database or matching events exist, the command prints
`{"status": "no_matching_events"}` without creating an MLflow run.

By default, one snapshot contains all currently matching events not yet
successfully uploaded by an incremental export to that tracking URI.
Both successful predictions and logged prediction failures are included.

New events arriving after snapshot creation wait for the next batch.

The result prints `batch_id`, `run_id`, `event_count`, and `artifact_uri`.
In MLflow UI, open the `predictive-maintenance-predictions` experiment,
the indicated run, and `predictions/predictions.json` under Artifacts.

Runs are tagged with `operation=predict`, `batch_id`, event count,
model URI information, and export mode.

The JSON contains batch metadata, selection filters,
and the complete original event records.

The local default database is separate from the Docker named volume.
To export the events stored by the Compose API, run inside its container:

```powershell
docker compose exec -T api python /app/scripts/export_predictions.py
```

The container uses the configured prediction database and internal MLflow URI.

### Historical snapshots and downloads

All original events stay in SQLite.

To explicitly upload a new snapshot of previously uploaded data,
use `--all`. Each successful `--all` invocation creates a new snapshot,
even for the same selection:

```powershell
python scripts/export_predictions.py --all
python scripts/export_predictions.py --all --since "2026-11-01T00:00:00Z" --until "2026-12-01T00:00:00Z"
python scripts/export_predictions.py --all --model-uri "models:/predictive-maintenance-model/2"
```

Replace the example model URI with the exact URI stored in the desired events.

Date filters use event timestamps, include `--since`, exclude `--until`,
and require an explicit timezone (`Z` means UTC).

These filters can also narrow an incremental export when `--all` is omitted.
Events without a timestamp are excluded when date filters are used.
Invalid saved timestamps produce an error.

Historical exports do not consume or reset the incremental queue.

Downloading an earlier artifact from MLflow UI is independent of these
commands and does not change export bookkeeping.

Nothing is deleted by export or download.

### Interrupted exports and concurrent use

SQLite records a stable batch ID, immutable event snapshot, destination,
and remote run ID.

After an upload failure, rerun the same command and selection.
The exporter resumes the pending batch and the same remote artifact path.

If remote run creation succeeded but its response was lost,
the stored batch tag is used to locate the run.

If the remote run finished before local completion was recorded,
the retry completes local bookkeeping without uploading the artifact again.

Conflicting duplicate remote batch tags are reported for manual resolution.

Complete an unfinished batch with its original filters before changing
the selection for the same destination.

Use a consistent tracking URI string;
different URI strings have separate incremental bookkeeping.

Export commands for the same local database are serialized with a separate
SQLite lock file (`predictions.db.export-lock` by default).
This lock does not block prediction-event writes during network uploads.

A second exporter may report a lock error;
retry after the first command finishes.

Keep the database and its export tables.
Deleting local bookkeeping loses the record of which batches were completed.

No upload scheduler is installed.
Export is explicitly started by the command.
Only JSON is generated; CSV is not required for the current Task 4.1 artifact.

## Monitoring actual API prediction logs

The monitoring command reads successful prediction events from SQLite
and publishes a monitoring state and, when enough samples exist,
an Evidently report.

In Compose:

```powershell
docker compose exec -T api python /app/scripts/run_api_monitoring.py --since "2026-11-23T00:00:00Z" --until "2026-11-24T00:00:00Z"
```

Replace the example timestamps with the desired window.
The start is inclusive and the end is exclusive.
Both timestamps must include a timezone.

The default minimum is 100 valid samples.

- Fewer samples produce `insufficient_data` and an unknown drift decision.
- Enough samples produce `ready`, a boolean drift decision and report files.
- Operational failures produce a nonzero exit code.

The command writes reports to `/app/reports/monitoring`.
The dashboard reads the same directory from the shared report volume.

Click **Refresh monitoring data** to display the latest published result.
The refresh button does not start monitoring or training.

Monitoring reads the SQLite event database directly.
Exporting events to MLflow is a separate operation and is not required first.

Manual verification covered:

- One real API prediction producing `insufficient_data`.
- A separate window of 100 real API requests producing a report.
- Matching counts and drift status in the dashboard.
- An embedded Evidently report and a visible HTML download button.

The 100-request check repeated one input as synthetic test traffic.
Its detected drift verifies the integration;
it does not establish drift in representative production traffic.

For local operation, configuration, report publication and the deterministic
normal/shifted demo, see the
[dashboard and monitoring guide](docs/dashboard_monitoring.md).

## Available endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Check API responsiveness; returns `{"status": "ok"}` |
| POST | `/train` | Train with `"v1"` or `"v2"`, register and conditionally select a champion |
| POST | `/predict` | Predict using the current champion and record its concrete version |

Successful requests return HTTP `200`.
Invalid request bodies return `422`.

Handled operational failures return `503` or `500`;
see the [API contract](docs/api_contract.md) for exact fields and messages.

## Running tests and quality checks

The focused API and service tests do not require the real CSV,
an existing champion, or a running MLflow server.
They use controlled service/model results and temporary synthetic data.

```powershell
python -m pytest tests/unit/test_api.py tests/unit/test_api_services.py tests/unit/test_api_schemas.py tests/unit/test_prediction_store.py -v
```

Run API/Registry integration tests with real temporary MLflow storage:

```powershell
python -m pytest tests/integration/test_api_registry_integration.py -v
```

These tests exercise model artifacts, alias changes, selection and
prediction events.

They replace the DVC training entrypoint with controlled training data;
real DVC restoration is covered by separate lineage tests.

Run focused monitoring command and dashboard-loader tests:

```powershell
python -m pytest tests/unit/test_run_api_monitoring.py tests/unit/test_dashboard_data.py -v
```

Before running the complete suite, restore the tracked datasets:

```powershell
python -m dvc pull -r dagshub
python -m pytest -v
```

MLflow integration tests use isolated temporary storage.
Dataset integration tests require the real datasets.

Check code style and types:

```powershell
python -m flake8 src tests infra app scripts
python -m pyright
```

The shared Pyright configuration references `.venv`.
If that directory is absent, Pyright may print an environment message.
Resolve actual errors and make sure it uses the intended Python interpreter.

## Current scope and remaining work

The API integrates verified DVC training with dataset versions `"v1"` and `"v2"`,
Registry registration and audited champion selection.

Prediction resolves the champion per request and caches concrete model versions.
Responses and persistent events identify the model actually used.
Prediction export and historical snapshots remain available.

Docker Compose runs the API, MLflow and the monitoring dashboard with
persistent storage. Its preparation service restores and verifies the raw
dataset and the `"v2"` training file.

The monitoring command reads actual API prediction events and publishes
reports consumed by the dashboard.

Manual insufficient-data and report-generation scenarios were verified.
The dashboard also rejects inconsistent drift decisions instead of showing
a misleading status.

Scheduled monitoring and drift-triggered automatic retraining remain
subsequent integration tasks.

The dashboard currently displays the latest valid training-selection audit.
It does not prove that a distinct automatic retraining job occurred.

Unit and isolated Registry integration tests cover promotion, rejection,
alias changes, missing champions, backend errors and consistent event versions.
API and service tests also cover dataset-version selection and validation.

Passing focused tests does not establish full-project CI or deployment status.
Review CI and the relevant runtime results before merging.

## Component documentation

- [API contract](docs/api_contract.md)
- [Monitoring documentation](docs/monitoring.md)
- [Dashboard and monitoring guide](docs/dashboard_monitoring.md)
- [DVC–MLflow integration guide](docs/dvc_mlflow_lineage.md)
- [Dataset release and export guide](docs/dataset_release.md)
- [Docker Compose instructions](docs/docker.md)
