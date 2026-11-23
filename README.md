# MLSecOps: Predictive Maintenance

A reproducible machine-learning system for predicting machine failure from six
operating measurements. The solution combines versioned data, a fitted
preprocessing/classification pipeline, MLflow tracking and Model Registry,
FastAPI, persistent prediction logging, Evidently monitoring, Streamlit and
optional drift-triggered retraining.

This README is the main project documentation and reproduction guide.
Detailed component documents are linked below.
Some older branch handoff documents describe intermediate designs; the current
code and this runbook describe the integrated behavior.

## Contents

- [Architecture and project structure](#architecture-and-project-structure)
- [Data and machine-learning pipeline](#data-and-machine-learning-pipeline)
- [Setup and access requirements](#setup-and-access-requirements)
- [API and MLflow](#api-and-mlflow)
- [Reproduce the complete system verification](#reproduce-the-complete-system-verification)
- [Automated tests, coverage and code quality](#automated-tests-coverage-and-code-quality)
- [GitHub Actions and Git workflow](#github-actions-and-git-workflow)
- [Screenshot evidence](#screenshot-evidence)
- [Submission dataset](#submission-dataset)
- [Limitations and AI assistance](#limitations-and-ai-assistance)
- [Detailed documentation](#detailed-documentation)

## Architecture and project structure

```text
Versioned CSV + Git/DVC provenance
    -> preprocessing + training -> MLflow run + registered candidate
    -> validation-based selection -> champion alias
                                      |
POST /predict -> concrete champion -> SQLite prediction events
                                      |                  |
                                      |                  -> MLflow JSON export
                                      -> Evidently report -> Streamlit
                                             |
                                  optional retraining on verified v2
                                             |
                                  candidate + selection -> champion
```

```text
MlSecOps_project/
├── app/
│   └── streamlit_dashboard.py       # Thin dashboard rendering layer
├── data/                            # Dataset contents and DVC pointers
│   ├── raw.dvc
│   ├── raw/                         # Original CSV, restored through DVC
│   └── processed/
│       ├── training/                # train_v1.csv.dvc, train_v2.csv.dvc
│       └── submission.dvc           # Preprocessed submission export
├── src/predictive_maintenance/
│   ├── data/                        # Validation, splits, preprocessing, provenance
│   ├── models/                      # Training, metrics, MLflow, Registry, selection
│   ├── api/                         # FastAPI routes, schemas and services
│   └── monitoring/                  # Events, export, drift, dashboard data, retraining
├── scripts/                         # CLI entry points and Compose verification
├── tests/
│   ├── unit/
│   └── integration/
├── docs/                            # Detailed component guides and evidence records
├── infra/                           # Placeholder for the future
├── screenshots/
│   ├── api/
│   ├── mlflow/
│   ├── docker/
│   └── monitoring/
├── .github/workflows/python-qa.yml
├── .dvc/config
├── .env.example
├── .coveragerc
├── .flake8
├── .pre-commit-config.yaml
├── pyrightconfig.json
├── pytest.ini
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

Root-level `data/` contains datasets. Python data modules live under
`src/predictive_maintenance/data/`; a reference such as `data/loader.py` in an
older handoff means that package directory, not root-level `data/`.

| Component | Main modules | Responsibility |
| --- | --- | --- |
| Data preparation | `data/loader.py`, `preprocessing.py`, `splitting.py`, `imbalance.py` | Validate the CSV, select features, create reproducible partitions and train-only transformations; calculate class weights. |
| Provenance and delivery | `data/provenance.py`, `split_manifest.py`, `submission_export.py` | Verify Git/DVC inputs, identify split membership and export the preprocessed submission data. |
| Training and Registry | `models/dvc_training.py`, `training.py`, `mlflow_tracking.py`, `training_registry.py`, `model_selection.py` | Train a complete pipeline, log artifacts/metrics/lineage, register a candidate and decide promotion. |
| API | `api/main.py`, `routes.py`, `schemas.py`, `services.py` | Validate requests, run real training, resolve/cache concrete champion versions and persist prediction events. |
| Events and export | `monitoring/prediction_store.py`, `prediction_export.py` | Store requests/results in SQLite and export recoverable JSON snapshots to MLflow. |
| Drift and dashboard data | `monitoring_pipeline.py`, `monitoring_reference.py`, `drift_report.py`, `summary.py`, `dashboard_data.py`, `dashboard_operational_data.py` | Build verified reference data, monitor successful events and supply validated report/Registry state to Streamlit. |
| Automation | `monitoring/automation.py`, `retraining.py`, `retraining_store.py`; `scripts/run_monitoring_worker.py` | Schedule cycles, check retraining eligibility, reserve inputs persistently and record training outcomes. |

Module paths in this table are relative to `src/predictive_maintenance/` unless
they explicitly start with `scripts/`. Mock helpers remain for isolated tests;
the deployed `/train` and `/predict` routes use real services.

## Data and machine-learning pipeline

The canonical input is the 5,000-row
`data/raw/predictive_maintenance_5000.csv`. The binary target is `Target`:
`1` indicates machine failure and `0` indicates no failure.

The six API/model inputs are product `type` (`L`, `M`, `H`), air and process
temperature in kelvin, rotational speed in rpm, torque in Nm and tool wear
in minutes. Identifiers and `Failure Type` are excluded from model features.

Preprocessing normalizes column names, adds process-minus-air temperature,
imputes numeric values with training medians and categorical values with the
training mode, scales numeric features and one-hot encodes product type.
The transformer is fitted only on training data and saved inside the sklearn
pipeline. Validation, future, test and prediction inputs use that fitted
transformer. Class imbalance is handled with balanced training sample weights.

| Partition | Initial v1 rows | Purpose |
| --- | ---: | --- |
| Train | 3,000 | Fit preprocessing and classifier. |
| Validation | 750 | Compare candidates and select the champion. |
| Future | 750 | Monitoring/demo observations and a labeled retraining reserve. |
| Final test | 500 | Reserved held-out evaluation; excluded from model selection. |

The split is disjoint, stratified and reproducible with seed 42.
The supported v2 training set adds ten explicitly selected, labeled future
records: 3,010 training rows and 740 remaining future rows. Validation and final
test remain unchanged. The v2 CSV must match this defined split; it is not an
arbitrary uploaded training file.

The supported classifiers are Gradient Boosting (default) and XGBoost.
MLflow records accuracy, precision, recall, F1, ROC AUC and average precision
on validation data, model parameters, fitted pipeline, input example and
Git/DVC/split lineage.

A candidate becomes `predictive-maintenance-model@champion` only if recall is
at least **0.70** and its validation F1 is strictly greater than the current
champion's F1. The first champion must also satisfy the recall threshold.
Rejected candidates remain traceable; equal or worse candidates do not replace
the champion. Dataset versions `v1`/`v2` and numeric Registry versions are
different identifiers.

## Setup and access requirements

### Required access and software

- Read access to the [GitHub repository](https://github.com/aszenas/MlSecOps_project.git).
- A DagsHub account/token with read access to the
  [shared data storage project](https://dagshub.com/bsumicz/MlSecOps_project).
  GitHub access alone does not grant access to these dataset objects.
  Ask the maintainers to grant DagsHub access if downloads are denied.
- Git and a full clone containing `.git/`. A GitHub source ZIP alone is
  insufficient for the Compose preparation/provenance checks.
- Docker Desktop running Linux containers, with Docker Compose v2.
  On Windows, enable the required WSL2/virtualization features.
- Python **3.11** for local verification/development; Docker and CI use 3.11.
  A host Python installation is not needed just to open the containerized UIs.
- Network access for image/package installation and DVC downloads; free local
  ports 8000, 5000 and 8501.

DagsHub is used for DVC object storage. MLflow runs locally inside Compose;
no separate hosted MLflow account is required. DVC uses remote `dagshub`,
S3 URL `s3://dvc`, and the endpoint in `.dvc/config`.

### Clone and start the full system

The following commands use PowerShell and run from the repository root:

```powershell
git clone https://github.com/aszenas/MlSecOps_project.git
cd MlSecOps_project
Copy-Item .env.example .env
```

Edit the ignored `.env` file locally:

```dotenv
DAGSHUB_TOKEN=<your-personal-token-with-data-access>
MONITORING_INTERVAL_SECONDS=300
MONITORING_WINDOW_SECONDS=3600
MONITORING_MINIMUM_SAMPLES=100
AUTOMATIC_RETRAINING_ENABLED=false #can be switched to true, false is the default value
```

Replace the token placeholder. Do not commit `.env`, tokens or
`.dvc/config.local`. Compose passes the token to the preparation service at
runtime, as both S3 access-key and secret-key values.

After configuration, the full stack starts with one command:

```powershell
docker compose up --build -d
docker compose ps -a
```

All services share `mlsecops-api:compose`. For later controlled rebuilds, the
same image can be built once with `docker compose build api`, followed by
`docker compose up -d --no-build`.

| Service | Role | Expected state / address |
| --- | --- | --- |
| `prepare` | Clone committed source into the training volume; pull and verify raw and v2 DVC data. | `Exited (0)` is success. |
| `mlflow` | Tracking server, Registry, SQLite metadata and model artifacts. | Healthy; [MLflow](http://127.0.0.1:5000). |
| `api` | Real training and champion prediction; persistent event logging. | Healthy; [Swagger](http://127.0.0.1:8000/docs). |
| `streamlit` | Read monitoring files and Registry/audit metadata. | Healthy; [dashboard](http://127.0.0.1:8501). |
| `monitoring` | Periodic monitoring/export, optional retraining. | Up; no HTTP health check. |

Preparation downloads `data/raw.dvc` and
`data/processed/training/train_v2.csv.dvc`. API startup waits for successful
preparation and healthy MLflow; dashboard and worker wait for the API.
Initial builds/downloads and Python startup can take several minutes.

| Named volume | Persistent contents |
| --- | --- |
| `mlflow_data` | Tracking/Registry database and model artifacts. |
| `prediction_data` | Prediction events, export batches and retraining reservations. |
| `training_workspace` | Clean committed source checkout and DVC data/cache. |
| `reports_data` | Monitoring state and Evidently report packages. |

`docker compose down` preserves these volumes. **Do not use `down -v` when
preserving results.** The prepared source is the host repository's committed
HEAD, not uncommitted edits. After source/data-pointer changes are committed,
stop the stack, rebuild if needed, and start it again to refresh preparation.

For startup errors, inspect `docker compose logs --tail=100 prepare mlflow api`.
A previous dependency timeout may be followed by a healthy service; check
`docker compose ps -a` and retry `docker compose up -d --no-build`.
The detailed [Docker guide](docs/docker.md) covers health diagnostics.

### Local setup (PowerShell)

Install local tools for the checks below:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install "pre-commit==4.7.0"
python -m pip check
python -m pre_commit install
```

If activation is unavailable, use `.\.venv\Scripts\python.exe` in place of
`python`. For local tests/export, restore data into the host clone too:
Compose's training-volume files are separate.

Configure DVC credentials locally without putting a token in command history:

```powershell
$dagshubCredential = Get-Credential -UserName "token" -Message "Enter your DagsHub token in the Password field"
python -m dvc remote modify --local dagshub access_key_id ($dagshubCredential.GetNetworkCredential().Password)
python -m dvc remote modify --local dagshub secret_access_key ($dagshubCredential.GetNetworkCredential().Password)
Remove-Variable dagshubCredential
python -m dvc pull -r dagshub
python -m dvc status
```

This stores credentials in ignored `.dvc/config.local`.
Standalone API/dashboard development commands are in the
[API contract](docs/api_contract.md) and
[dashboard guide](docs/dashboard_monitoring.md).

## API and MLflow

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Returns `{"status":"ok"}`; checks API responsiveness, not model readiness. |
| `POST /train` | Verify data/provenance, fit a pipeline, log/register a candidate and select the champion. |
| `POST /predict` | Resolve the champion alias, run the fitted pipeline and persist the request/result. |

In Swagger, select a route, click **Try it out**, enter JSON and click
**Execute**. Example training request:

```json
{"dataset_version": "v2", "model_type": "gradient_boosting"}
```

An empty object uses `v1`, Gradient Boosting and seed 42. Only dataset ID
`default` and seed 42 are supported. The training response includes
`training_run_id`, `model_uri`, metrics, `registered_version`,
`champion_version`, `promoted` and `selection_reason`.
HTTP 200 / `completed` means training/selection completed, not necessarily
promotion. The response does not contain a `dataset_version` field.

Example prediction request:

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

The response contains class `prediction`, failure `probability`, concrete
Registry `model_version` and `mode: "real"`. Probability >= 0.5 maps to class 1.
The API resolves the champion for each request and caches the concrete
model-URI/backend pair, so promotion is observed without restarting the API.
Without a champion, prediction returns a controlled 503; invalid payloads
return 422.

Validated prediction requests/results, including handled service errors, are
stored in SQLite with a UTC timestamp and `X-Request-ID`. Successful responses
require successful event persistence. Schema-invalid 422 requests do not enter
this event log.

Training runs appear in `predictive-maintenance-baselines`. Prediction snapshots
appear in `predictive-maintenance-predictions`, with artifact
`predictions/predictions.json`. Export is asynchronous to the request:

```powershell
docker compose exec -T api python /app/scripts/export_predictions.py
```

Default export uploads pending events, preserves source records, and resumes
an interrupted batch. `--all` creates a historical snapshot, optionally filtered
with `--since`, `--until` or `--model-uri`. Downloading an existing MLflow artifact
does not change export bookkeeping.

## Reproduce the complete system verification

Use the following order so automatic promotion cannot interfere with the
persistence snapshot. These checks exercise the running stack; they are separate
from the automated pytest suite.

### 1. Start with automatic retraining disabled

Keep `AUTOMATIC_RETRAINING_ENABLED=false` in `.env`, start Compose as above,
and wait for the expected service states. Scheduled monitoring/export can still
run; the flag controls training only.

### 2. Verify the real API–MLflow integration

With the host virtual environment active:

```powershell
python scripts/check_compose.py
```

Expected: API, MLflow and Streamlit are OK, a real default-v1 training operation
completes, a champion prediction succeeds, and the script ends with
`Compose check: OK`. It saves run/version/request IDs in ignored
`artifacts/compose-check.json`. Training/prediction also remain callable in
Swagger for screenshots.

### 3. Verify persistence across container recreation

Do not run additional training or change the champion between these checks.

```powershell
docker compose down
docker compose up -d --no-build
# Wait until API, MLflow and Streamlit are healthy.
python scripts/check_compose.py --verify-persistence
```

Expected: the saved run and Registry versions exist, model artifacts download,
the saved event is unchanged, and champion serving still works:
`Persistence check: OK`. The script explicitly sets tracking and Registry
URIs to the Docker-published MLflow address for artifact retrieval.

### 4. Verify insufficient data, then 100 successful predictions

Pause the scheduled worker to keep it from replacing the manually selected
dashboard window:

```powershell
docker compose stop monitoring
$predictionBody = @{
    type = "L"
    air_temperature_k = 300.8
    process_temperature_k = 310.3
    rotational_speed_rpm = 1538
    torque_nm = 36.1
    tool_wear_min = 198
} | ConvertTo-Json

$monitoringSince = [DateTimeOffset]::UtcNow.ToString("o")
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $predictionBody -TimeoutSec 60 -ErrorAction Stop
$monitoringUntil = [DateTimeOffset]::UtcNow.ToString("o")
docker compose exec -T api python /app/scripts/run_api_monitoring.py --since $monitoringSince --until $monitoringUntil
```

Click **Refresh monitoring data** in Streamlit. With no other requests in that
window, expect `insufficient_data`, 1 valid sample, 99 missing samples and
unknown drift (`null`), without a current report.

Create a new window containing 100 successful real API requests:

```powershell
$monitoringSince = [DateTimeOffset]::UtcNow.ToString("o")
foreach ($i in 1..100) {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $predictionBody -TimeoutSec 60 -ErrorAction Stop | Out-Null
}
$monitoringUntil = [DateTimeOffset]::UtcNow.ToString("o")
docker compose exec -T api python /app/scripts/run_api_monitoring.py --since $monitoringSince --until $monitoringUntil
```

Stop if a request fails. Refresh Streamlit: expect `ready`, 100 valid samples,
a boolean drift decision, the embedded Evidently report and
**Download Evidently HTML report**. Exactly 100 assumes no concurrent requests.
The repeated identical input is synthetic demonstration traffic, not evidence
of drift in representative production observations.

Monitoring compares the six input features against the initial 3,000-row
training reference. Numeric drift uses Wasserstein distance and categorical
drift uses Jensen–Shannon distance, each with threshold 0.1. Dataset drift is
reported when at least 50% of monitored fields drift. Reports are written as
HTML, JSON and summary JSON under `reports/monitoring/runs/<id>/`; current state
is published atomically in `monitoring_state.json`.

### 5. Enable and verify automatic retraining

Set `AUTOMATIC_RETRAINING_ENABLED=true` in `.env`. With the background worker
still stopped, run one controlled cycle using the same persistent volumes:

```powershell
docker compose run --rm --no-deps -e AUTOMATIC_RETRAINING_ENABLED=true monitoring python /workspace/repository/scripts/run_monitoring_worker.py --once
```

This uses the latest rolling window (default one hour); run it while the
100 predictions are still within that window. It requires drift and verified
labeled v2 data. It does not use model predictions as target labels.

| Retraining result | Interpretation |
| --- | --- |
| `disabled` | Training switch is off. |
| `insufficient_data` / `no_drift` | Monitoring does not justify a training attempt. |
| `waiting_for_labeled_data` | The required v2 CSV or DVC pointer is absent. |
| `completed` | Candidate training/selection completed; inspect `training.selection` for promotion. |
| `duplicate_input` | This dataset/backend/model combination was already reserved; no new training starts. |
| `failed` | Verification/training encountered an error; inspect details. |

A new successful automatic run is marked in MLflow with
`operation=automated_retraining`, `retraining_attempt_id`,
`retraining_dataset_version=v2` and monitoring-window tags.
The dashboard shows the latest valid selection audit and automatic-run marker;
worker waiting/duplicate states are read from logs, not that audit panel.

Observed Docker result on 2026-11-23: run
`a6a0b5bc203244739d3265ea31ef2a95` completed, created candidate **3** and
retained champion **2**, because both validation F1 scores were
`0.6538461538461539` (`f1_not_improved`). Repeating the cycle returned
`duplicate_input`, demonstrating persistent duplicate protection. These IDs
describe the recorded environment; a fresh installation produces different IDs.

Resume normal scheduling with the new environment setting:

```powershell
docker compose up -d --no-build --force-recreate monitoring
docker compose logs --tail=30 -f monitoring
```

Ctrl+C exits log following without stopping the service. The worker runs a
cycle immediately and then waits 300 seconds by default. A completed cycle
does not always train: eligibility and duplicate protection still apply.

**Current data boundary:** retraining explicitly supports v2, not automatic v3
discovery. A reservation is unique by raw/training DVC hashes, MLflow URI and
model type; another window or application commit does not create fresh input.
Even failed/interrupted attempts remain reserved pending manual reconciliation.
Do not delete the event database to force another run. Docker evidence covers
successful training with rejection and duplicate protection; promotion and
training-failure serving are additionally covered by isolated integration tests.

## Automated tests, coverage and code quality

### Test layers

| Layer | Location and examples | What is verified |
| --- | --- | --- |
| Unit | `tests/unit/test_loader.py`, `test_preprocessing.py`, `test_splitting.py`, `test_provenance.py`, `test_submission_export.py` | Schema/errors, train-only preprocessing, deterministic disjoint splits, DVC/Git checks and export contracts. |
| Unit | `test_api.py`, `test_api_services.py`, `test_model_selection.py`, `test_training_registry.py` | API validation/status codes, service wiring, champion cache/version behavior and promotion/rejection rules with isolated dependencies. |
| Unit | `test_prediction_store.py`, `test_prediction_export.py`, `test_monitoring_automation.py`, `test_retraining*.py`, `test_monitoring_worker.py` | Event persistence, recoverable batches, eligibility, persistent claims, retries and worker control. |
| Dashboard | `test_dashboard_data.py`, `test_dashboard_operational_data.py`, `test_streamlit_dashboard.py` | Missing/invalid states, report/Registry loading, Streamlit rendering and download visibility. |
| Integration: data/models | `test_data_pipeline.py`, `test_model_training.py`, `test_submission_export_integration.py` | Real CSV through preprocessing/training, persisted pipeline inference, export membership and leakage checks. |
| Integration: MLflow/Registry | `test_mlflow_integration.py`, `test_model_registry_integration.py`, `test_training_registry_integration.py` | Real temporary MLflow stores, logged/reloaded artifacts, aliases, selection, provenance and v2 evaluation-partition preservation. |
| Integration: API/monitoring | `test_api_registry_integration.py`, `test_monitoring_report.py`, `test_persisted_monitoring.py` | Real Registry/API events, report generation, promotion/rejection, drift-triggered orchestration and failure/duplicate protection. |

Unit tests isolate external operations with temporary data and mocks.
Integration tests use actual libraries/artifacts and isolated SQLite stores;
selected API/retraining tests use controlled training/reference adapters to
make promotion, rejection and failure deterministic. They do not all call the
live DagsHub remote or running Docker services. Compose checks above provide
the separate live-system verification.

### Run the same test groups and gates locally

Restore the raw DVC dataset first. From the repository root, with the venv active:

```powershell
python -m coverage erase
python -m coverage run -m pytest tests/unit/ -v
python -m coverage run --append -m pytest tests/integration/test_mlflow_integration.py -v
python -m coverage run --append -m pytest tests/integration/ --ignore=tests/integration/test_mlflow_integration.py -v
python -m coverage report -m --include="src/*" --fail-under=95
python -m coverage report -m --include="src/*,tests/*" --fail-under=95
python -m coverage xml -o reports/coverage.xml
python -m coverage html -d reports/coverage-html
python -m flake8 src tests infra app scripts
python -m pyright
python -m pre_commit run --all-files
```

Continue only after each required command succeeds. A focused test subset does
not establish whole-project coverage. Review the current GitHub Actions report
for the final tested revision rather than treating historical test counts as
current results.

| Check | Actual configuration |
| --- | --- |
| Coverage | `.coveragerc`: statement/line coverage, not branch coverage. Sources include `src`, `tests`, `infra`, `app`. CI separately enforces >=95% for `src/*` and combined `src/*,tests/*`. Test files therefore count too. |
| PEP8 / complexity | `.flake8`: maximum line length 120 and complexity 10; CI fails on reported violations, stricter than accepting a 1% error allowance. |
| Types | `pyrightconfig.json`: standard mode, Python 3.11; checks `src`, `tests`, `infra`, `app`, `scripts`, with zero-error expectation. Strict-mode bonus is not claimed. |
| Pre-commit | Whitespace, final newline, YAML, conflict markers, added files over 1 MiB and Flake8. Tests/coverage run in the separate QA job. |

Coverage exclusions target environments, dependencies, caches and generated
build/report directories; no application-module exclusion is configured.
The explicit CI gates above define their scopes; an overall percentage alone
does not establish both required results. Inspect uncovered lines in the
text/HTML reports. If a pre-commit hook fixes a file, review/stage the fix and
rerun it before committing.

## GitHub Actions and Git workflow

One workflow, [Python Quality Assurance](.github/workflows/python-qa.yml),
runs on pull requests targeting `dev`/`main`, pushes to those branches and
manual dispatch. It contains three jobs:

| Job | Checks and outputs |
| --- | --- |
| **Python QA** | Ubuntu/Python 3.11, dependency installation and `pip check`, ML import smoke check, authenticated raw DVC restore, unit tests, MLflow integration tests, remaining integration tests, both 95% coverage gates, Flake8 and Pyright. |
| **Pre-commit checks** | Runs all configured hooks against tracked files and displays required corrections. |
| **Compose validation and image build** | Validates `docker-compose.yaml` and builds images. Uses a placeholder token for configuration validation; it does not launch the complete stack or prove runtime DVC access. |

Python QA needs a repository Actions secret named `DAGSHUB_TOKEN` with access to
the shared data. It maps that secret to both AWS credential variables for the
DagsHub S3 endpoint. Missing credentials/downloads fail QA; dataset-dependent
integration tests are not silently omitted. Secrets may be unavailable to
pull requests from forks.

The QA artifact `python-qa-reports-<run-id>` (retention: 14 days) contains
JUnit XML for all three test groups, source/combined coverage text,
`coverage.xml` and browsable coverage HTML. The Actions job summary reports
test/gate outcomes and labels coverage **complete** only when all three test
groups passed; otherwise it is **partial**. CI configuration documents intended
checks, while the Actions run for the submitted commit is the evidence of
their actual result.

Development uses `feature/*` and `fix/*` branches, reviewed PRs into `dev`,
and a reviewed release PR into `main`. Merge commits preserve branch history.
Review notes, approvals and frequent meaningful commits remain in GitHub/Git;
this README describes component behavior rather than individual contributions.
See [Git workflow](docs/branching.md).

## Screenshot evidence

Paths below use the actual repository filenames, including their existing
spelling. Numbered groups are file prefixes, not additional subdirectories.
Run/version IDs in screenshots belong to the recorded run.

| Folder / files | Evidence |
| --- | --- |
| [`screenshots/api/`](screenshots/api), `01_train_1.png`–`01_train_3.png` | Swagger training request/response and real training/Registry selection. |
| `screenshots/api/02_predict_1.png`–`02_predict_3.png` | Real prediction endpoint, probability and served model version. |
| [`screenshots/mlflow/`](screenshots/mlflow), `01_v2_validation_metrics_1.png`–`01_v2_validation_metrics_3.png` | v2 run metrics in MLflow. |
| `screenshots/mlflow/02_v2_promotion_decision_1.png`–`02_v2_promotion_decision_3.png` | Recorded candidate/champion selection and promotion decision. |
| `screenshots/mlflow/03_v2_dataset_lineage_1.png`–`03_v2_dataset_lineage_2.png` | Versioned dataset/Git lineage and split information. |
| `screenshots/mlflow/04_registry_champion_1.png`–`04_registry_champion_2.png` | Model Registry versions and the champion alias on the selected model. |
| [`screenshots/docker/01-compose-services.png`](screenshots/docker/01-compose-services.png) | Compose services and expected running/completed states. |
| [`screenshots/docker/02-verify-persitence.png`](screenshots/docker/02-verify-persitence.png) | Successful persistence verification after container recreation. |
| [`screenshots/monitoring/`](screenshots/monitoring), `01-insufficiant_data-prediction1.png`, `01-insufficiant_data-prediction2.png` | Insufficient-data scenario with one valid prediction and 99 missing samples. |
| `screenshots/monitoring/02-100-successful-prediction1.png`, `02-100-successful-prediction2.png` | The 100-successful-prediction scenario and report-ready state. |
| `screenshots/monitoring/03-streamlitdashboard1.png`–`03-streamlitdashboard8.png` | Streamlit monitoring overview, drift details and embedded Evidently report views. |
| `screenshots/monitoring/04-automated-retraining1.png`, `04-automated-retraining2.png` | Finished automatic run, metrics, `operation=automated_retraining`, attempt/data tags and retained-champion selection. |
| [`screenshots/monitoring/05-automated-retraining-duplicaed-input.png`](screenshots/monitoring/05-automated-retraining-duplicaed-input.png) | A repeated input is blocked with `duplicate_input`. |

The detailed [verification record](docs/verification_record.md) also records
submission restoration hashes and the automatic-training run IDs. Screenshot
files must be committed with the submission; local untracked images are not
visible to a reviewer.

## Submission dataset

The assignment's preprocessed dataset is separately versioned by
`data/processed/submission.dvc`. Restore it with the local DVC credentials:

```powershell
python -m dvc pull data/processed/submission.dvc -r dagshub
python -m dvc status data/processed/submission.dvc
```

The output directory contains `submission_preprocessed.csv` and
`submission_metadata.json`. The CSV contains all 5,000 rows, nine transformed
features, `target`, `partition` and `source_udi` (12 columns). Partition counts
remain 3,000/750/750/500. It must not be treated as one training set because
validation and final-test rows are included.

To regenerate rather than download the export, restore raw data first, then:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/export_preprocessed_submission.py
```

The metadata records raw DVC lineage, split identity, partition counts and
dependency versions. A separate clean-clone restore verified the versioned
submission output and matching SHA-256 hashes; see the
[dataset release guide](docs/dataset_release.md).
That historical restore evidence concerns the submission output specifically.


## Limitations and AI assistance

- Automatic retraining currently uses the defined labeled v2 dataset. There
  is no generic v3 ingestion/discovery, target-label collection or automatic
  reconciliation/reset for failed/interrupted reservations.
- The dashboard reads results; refresh does not trigger monitoring or training.
  Reports/events have no automatic retention or cleanup policy.
- Drift thresholds are demonstration settings. Synthetic repeated requests
  demonstrate the pipeline, not production drift or guaranteed model gains.
- Airflow is not implemented; scheduling uses the Compose monitoring worker.
- Deployment is a local coursework stack with loopback-published ports.
  Authentication, TLS and multi-worker/distributed training coordination are
  not implemented, so this is not a hardened public production deployment.
- AI assistance supported planning, implementation suggestions, tests, debugging, documentation.
  Changes were reviewed and exercised through local checks, manual Compose
  scenarios and the configured CI process; AI-generated suggestions were
  checked against the actual code and execution results.

## Detailed documentation

- [API contract and errors](docs/api_contract.md)
- [Docker setup, volumes and troubleshooting](docs/docker.md)
- [Monitoring and drift detection](docs/monitoring.md)
- [Dashboard and manual API-to-report verification](docs/dashboard_monitoring.md)
- [Automated retraining](docs/automated_retraining.md)
- [DVC–MLflow lineage](docs/dvc_mlflow_lineage.md)
- [Dataset release and submission export](docs/dataset_release.md)
- [Verification record](docs/verification_record.md)
- [Git workflow](docs/branching.md)
- [Preprocessing design](docs/preprocessing_branch_documentation_en.md)
- [MLflow training design](docs/mlflow_training_branch_documentation_en.md)
- [Historical Registry design/handoff](docs/model_registry_branch_documentation_en.md)
