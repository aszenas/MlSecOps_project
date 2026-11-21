# Docker API image

## Scope

The root Dockerfile builds the FastAPI application using Python 3.11.
The image can start and serve `/health` without a dataset or model.
Training requires the dataset and an accessible MLflow backend.
Prediction requires MODEL_URI, an accessible model and MLflow backend,
and a writable prediction log location.
The complete Docker Compose environment is a separate integration task.

## Build

Run from the repository root with Docker Desktop running in Linux container mode:

```powershell
docker build --progress=plain -t mlsecops-api:dev .
```

The build installs the dependencies from requirements.txt and runs pip check.

## Run

```powershell
docker run -d --name mlsecops-api-check -p 127.0.0.1:8001:8000 mlsecops-api:dev
```

The API is available only on the local machine:

- Health: http://127.0.0.1:8001/health
- Swagger: http://127.0.0.1:8001/docs

Check the health endpoint after the application starts:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Expected health response:

```json
{"status": "ok"}
```

The health endpoint confirms that the API responds. It does not verify
model availability, dataset access, MLflow connectivity, or prediction storage.

View container logs if startup fails:

```powershell
docker logs mlsecops-api-check
```

## Cleanup

```powershell
docker stop mlsecops-api-check
docker rm mlsecops-api-check
```

The built image remains available.

## CI

The Python Quality Assurance workflow includes an independent Docker image
build job. It uses the root Dockerfile and does not require DagsHub credentials.

The job follows the workflow's existing event and branch filters.
It verifies image construction; it does not run an API health check.

## Validation

Locally verified on Windows using Docker Desktop with Linux containers:

- Image build completed successfully.
- Build-time pip check reported no broken requirements.
- The container started successfully.
- The health endpoint returned `{"status": "ok"}`.

GitHub Actions build validation must be confirmed separately.

## Compose integration handoff

The Compose configuration must provide dataset access, MLflow connectivity,
MODEL_URI, and persistent writable prediction storage.

The default prediction database is `/app/artifacts/predictions.db`.
The current image does not prepare a writable `/app/artifacts` directory.
Provide a directory writable by UID 10001 and configure PREDICTION_LOG_DB
if using a different path.

Container startup may take several seconds. Wait for application startup
before checking `/health`.
