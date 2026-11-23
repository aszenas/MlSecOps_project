# Docker Compose environment

## Scope

The stack provides FastAPI, MLflow and a temporary Streamlit page.
A one-shot preparation service creates a committed training checkout
and restores its DVC-managed dataset.

The real monitoring dashboard will replace the temporary Streamlit page
after the component handoff.

## Prerequisites

- Docker Desktop running with Linux containers.
- Project dependencies installed locally for the verification scripts.
- A personal DagsHub token with access to the shared dataset.
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

## Local addresses

- Swagger: http://127.0.0.1:8000/docs
- MLflow: http://127.0.0.1:5000
- Streamlit: http://127.0.0.1:8501

## Persistent storage

- `mlflow_data`: MLflow metadata, Registry records and artifacts.
- `prediction_data`: prediction events and export bookkeeping.
- `training_workspace`: committed training checkout and DVC data/cache.
- `reports_data`: shared monitoring reports.

The API can write reports; Streamlit mounts the report volume read-only.

The API imports source code from the prepared Git checkout.
Refresh that checkout with the stack stopped when changing source versions.

## Functional verification

Once the services are healthy:

```powershell
.\.venv\Scripts\python.exe scripts/check_compose.py
```

This creates a real training run and prediction.
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

## Troubleshooting

```powershell
docker compose logs --tail=100 prepare api mlflow streamlit
```

A failed preparation prevents API startup.
Check the preparation log for Git, DVC or authentication failures.

`docker compose down` preserves named volumes.
Adding `-v` removes the volumes and their stored data.

## Validation status

Image builds, service health checks and individual storage checks
were verified during development.

The complete functional and persistence checks must be recorded after
successful execution. CI results must be confirmed on the pull request.
