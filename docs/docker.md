# Docker Compose environment

## Scope

The stack provides FastAPI, MLflow and the Streamlit monitoring dashboard.

A one-shot preparation service creates a committed training checkout
and restores its DVC-managed raw dataset and `"v2"` training file.

The API writes prediction events to persistent SQLite storage.
The monitoring command publishes reports to the shared report volume,
which the dashboard reads without modifying it.

The `monitoring` service schedules monitoring and incremental prediction export.
Automatic retraining is optional and defaults to disabled. See the
[automated retraining guide](automated_retraining.md) for activation and checks.

## Prerequisites

- Docker Desktop running with Linux containers.
- Project dependencies installed locally for the verification scripts.
- A personal DagsHub token with access to the shared datasets.
- Committed application-code changes before preparing a new checkout.

## Configuration

Copy `.env.example` to `.env` and set `DAGSHUB_TOKEN`.
Keep `.env` outside Git.

The preparation service receives the token at runtime.
Credentials are not embedded in the Docker image.

The API and dashboard use the internal MLflow address:

```text
http://mlflow:5000
```

The dashboard uses:

```text
MONITORING_REPORT_DIR=/app/reports/monitoring
```

This directory is part of the shared `reports_data` volume.

## Startup

Run from the repository root:

```powershell
docker compose config --quiet
docker compose build api
docker compose up -d --no-build
docker compose ps -a
```

All services use the shared `mlsecops-api:compose` image.
Building `api` once prepares that image for the entire stack.

Expected states:

- `prepare`: Exited (0).
- `api`, `mlflow`, `streamlit`: healthy.
- `monitoring`: Up; it is a background worker without an HTTP health check.

The preparation service runs once and exits after verification.
Health checks indicate HTTP responsiveness.

Startup may take time while Python dependencies load and MLflow initializes.
The API waits for successful preparation and a healthy MLflow service.
Streamlit waits for a healthy API.

## Local addresses

- Swagger: http://127.0.0.1:8000/docs
- MLflow: http://127.0.0.1:5000
- Dashboard: http://127.0.0.1:8501

## Training dataset versions

The preparation service restores and verifies both DVC targets:

- `data/raw.dvc`
- `data/processed/training/train_v2.csv.dvc`

Both targets must be accessible in the configured DagsHub remote.
A failed download or verification prevents API startup.

The API defaults to `"v1"`, which builds the initial training partition
from the raw dataset.

To train with the expanded labeled dataset, send this request to `POST /train`:

```json
{
  "dataset_version": "v2"
}
```

Model type and random state use their existing defaults.
Both dataset versions preserve the original validation and final-test partitions.

The separate `train_v1.csv` export is not required by the current `"v1"`
training path.

Champion promotion remains conditional on validation results.
A completed `"v2"` training operation may keep the previous champion.

Dataset versions are separate from numeric MLflow Registry model versions.

## Persistent storage

| Volume | Contents |
| --- | --- |
| `mlflow_data` | MLflow metadata, Registry records and artifacts |
| `prediction_data` | Prediction events and export bookkeeping |
| `training_workspace` | Committed training checkout and DVC data/cache |
| `reports_data` | Monitoring state and report artifacts |

The API container writes monitoring reports.
Streamlit mounts the report volume read-only.

The API imports source code from the prepared Git checkout.
The preparation service updates that checkout to the source repository's
committed HEAD.

Refresh the checkout with the stack stopped when changing source versions.

After committing source changes:

```powershell
docker compose down
docker compose build api
docker compose up -d --no-build
```

Existing build layers are reused when their inputs have not changed.

`docker compose down` preserves named volumes.
Do not add `-v` when retaining models, prediction events and reports.

The container prediction database is separate from a database created
directly in the host repository.

## Functional verification

Once the services are healthy:

```powershell
.\.venv\Scripts\python.exe scripts/check_compose.py
```

This creates a real training run and prediction using the API's default
dataset version, `"v1"`.

Identifiers are saved to the Git-ignored `artifacts/compose-check.json`.

Then recreate containers while retaining their volumes:

```powershell
docker compose down
docker compose up -d --no-build
```

Once healthy, verify the saved records without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_compose.py --verify-persistence
```

The check verifies the run, Registry versions, model artifact download,
saved prediction event and continued champion serving.

Do not intentionally change the champion between saving the verification
snapshot and checking its persistence. Pause the monitoring worker during
this check if automatic retraining is enabled.

## Manual v2 verification

In Swagger UI, call `POST /train` with:

```json
{
  "dataset_version": "v2"
}
```

A successful operation returns HTTP `200` with `status: "completed"`,
a training run ID and the Registry selection result.

Open the returned run in MLflow and inspect:

- The dataset split version.
- The `dataset_training_dvc_hash` tag.
- The `lineage/split_manifest.json` artifact.
- The candidate metrics and selection decision.

The `promoted` field may be either `true` or `false`.
Successful execution does not guarantee a better candidate.

When a champion is available, call `POST /predict` and verify that its
`model_version` matches the selected `champion_version`.

Do not run this additional training between saving a functional-check snapshot
and verifying its persistence: promotion may intentionally change the champion.

## Monitoring API prediction events

Monitoring runs inside the API container and uses its configured database,
verified reference dataset and writable report volume.

```powershell
docker compose exec -T api python /app/scripts/run_api_monitoring.py --since "2026-11-23T00:00:00Z" --until "2026-11-24T00:00:00Z"
```

Replace the timestamps with the desired window.
The start is inclusive and the end is exclusive.
Both timestamps must include a timezone.

The command publishes its state at:

```text
/app/reports/monitoring/monitoring_state.json
```

Successful report packages are stored under:

```text
/app/reports/monitoring/runs/<run-id>/
```

The dashboard reads these files from the same persistent volume.
Click **Refresh monitoring data** after the command finishes.

### Expected outcomes

- Fewer than 100 valid samples: `insufficient_data`.
- At least 100 valid samples: `ready`, a boolean drift decision and reports.
- Missing database, invalid reference data or another handled operational
  failure: an error message and exit code 1.
- Invalid command arguments: exit code 2.

The dashboard refresh does not start monitoring.
The manual `run_api_monitoring.py` command does not start training.
The separate worker can retrain when automatic retraining is enabled.

For command options and detailed behavior, see the
[dashboard and monitoring guide](dashboard_monitoring.md).

## Prediction export inside Compose

To export the events stored by the Compose API to its MLflow backend:

```powershell
docker compose exec -T api python /app/scripts/export_predictions.py
```

The command uses the API container's `PREDICTION_LOG_DB`
and `MLFLOW_TRACKING_URI`.

This export is separate from drift monitoring.
The monitoring command reads persisted SQLite events directly.

## Troubleshooting

Inspect the relevant service logs:

```powershell
docker compose logs --tail=100 prepare api mlflow streamlit
```

A failed preparation prevents API startup.
Check the preparation log for Git, DVC or authentication failures.

If the `"v2"` download fails, verify that its data object has been uploaded
to the shared DagsHub remote and that the runtime token can access it.

If a service is unhealthy, inspect its health-check results:

```powershell
docker inspect mlsecops_project-mlflow-1 --format '{{json .State.Health}}'
docker inspect mlsecops_project-api-1 --format '{{json .State.Health}}'
```

The actual container-name prefix may differ if the Compose project name changes.

A service may finish startup after an earlier Compose dependency error.
If it is now healthy, retry startup without rebuilding:

```powershell
docker compose up -d --no-build
```

A successful HTTP response in the application log does not alone prove that
the health-check process finished within its configured timeout.

The MLflow health check allows extra startup time and a longer execution
timeout to accommodate slower local startup.

If a build fails with a pip download timeout, retry after checking connectivity.
The Dockerfile allows a longer pip socket timeout.

Do not remove volumes to resolve an ordinary startup or health-check timeout.
Adding `-v` to `docker compose down` deletes the stored volume data.

## Validation status

Image builds, service health checks and individual storage checks
were verified during development.

A functional check with real training and champion prediction was previously
observed.

The API-log monitoring integration was manually verified in Compose:

- One successful prediction produced `insufficient_data`.
- The dashboard showed one valid sample and 99 missing samples.
- A separate window of 100 successful API requests produced `ready`.
- No invalid event or feature rows were skipped in that window.
- The dashboard displayed the detected drift and embedded Evidently report.
- The HTML report download button was visible.
- The dashboard queried the current champion from the shared MLflow backend.

The 100-request verification used repeated identical input.
The drift result demonstrates the integration, not representative
production-data drift.

The preparation service also successfully restored and verified the `"v2"`
training file after dataset-version integration.

Complete persistence-check results must be recorded after successful execution.
CI results must be confirmed on the pull request.

The automated one-shot Docker run on 2026-11-23 trained candidate version 3,
recorded automatic-run tags and retained champion version 2 because F1 did not
improve. See the [verification record](verification_record.md#automated-retraining-verification).
