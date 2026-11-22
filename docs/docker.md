# Docker Compose environment

## Scope

The stack provides FastAPI, MLflow and a temporary Streamlit page.

A one-shot preparation service creates a committed training checkout
and restores its DVC-managed raw dataset and `"v2"` training file.

The real monitoring dashboard will replace the temporary Streamlit page
after the component handoff.

## Prerequisites

- Docker Desktop running with Linux containers.
- Project dependencies installed locally for the verification scripts.
- A personal DagsHub token with access to the shared datasets.
- Committed application-code changes before preparing a new checkout.

## Configuration

Copy `.env.example` to `.env` and set `DAGSHUB_TOKEN`.
Keep `.env` outside Git.

The preparation service receives the token at runtime.
The API uses the internal MLflow address `http://mlflow:5000`.

## Startup

Run from the repository root:

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps -a
```

Expected states:

- `prepare`: Exited (0).
- `api`, `mlflow`, `streamlit`: healthy.

The preparation service runs once and exits after verification.
Health checks indicate HTTP responsiveness.

MLflow startup can take time while database initialization and application
startup complete. Dependent services wait for its health check.

## Local addresses

- Swagger: http://127.0.0.1:8000/docs
- MLflow: http://127.0.0.1:5000
- Streamlit: http://127.0.0.1:8501

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

- `mlflow_data`: MLflow metadata, Registry records and artifacts.
- `prediction_data`: prediction events and export bookkeeping.
- `training_workspace`: committed training checkout and DVC data/cache.
- `reports_data`: shared monitoring reports.

The API can write reports; Streamlit mounts the report volume read-only.

The API imports source code from the prepared Git checkout.
Refresh that checkout with the stack stopped when changing source versions.

After committing changes, recreate the stack:

```powershell
docker compose down
docker compose up --build -d
```

The preparation service updates the training checkout to the source commit
and verifies the required data again.

Do not add `-v` when preserving existing models, events and reports.

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
docker compose up -d
```

Once healthy, verify the saved records without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/check_compose.py --verify-persistence
```

The check verifies the run, Registry versions, model artifact download,
saved prediction event and continued champion serving.

### Manual v2 verification

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

## Troubleshooting

```powershell
docker compose logs --tail=100 prepare api mlflow streamlit
```

A failed preparation prevents API startup.
Check the preparation log for Git, DVC or authentication failures.

If the `"v2"` download fails, verify that its data object has been uploaded
to the shared DagsHub remote and that the runtime token can access it.

If MLflow is unhealthy, inspect its startup logs and allow initialization
to finish before retrying startup.

`docker compose down` preserves named volumes.
Adding `-v` removes the volumes and their stored data.

## Validation status

Image builds, service health checks and individual storage checks
were verified during development.

A successful functional check with real training and champion prediction
was also observed during development.

The preparation service successfully restored and verified the `"v2"`
training file after dataset-version integration.

Full persistence-check results and the manual `"v2"` training result must
be recorded after successful execution.
CI results must be confirmed on the pull request.