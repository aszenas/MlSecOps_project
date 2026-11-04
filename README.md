# MlSecOps_project

Predictive Maintenance / MLSecOps project by Mate, Anna, and Beni.

## Project structure

```text
MlSecOps_project/
├── app/                         # Streamlit dashboard placeholder
├── data/                        # Dataset files, not Python modules
│   ├── raw/                     # Original CSV, managed separately from Git
│   ├── raw.dvc                  # DVC metadata for the raw dataset
│   └── processed/               # Processed dataset storage
├── docs/                        # API contract, workflow, and component handoffs
├── infra/                       # Infrastructure configuration placeholder
├── scripts/                     # Baseline training and model verification
├── src/
│   └── predictive_maintenance/
│       ├── api/                 # FastAPI routes, schemas, and API services
│       ├── data/                # Loading, preprocessing, and splitting code
│       ├── models/              # Training, evaluation, and MLflow tracking
│       └── monitoring/          # Monitoring package placeholder
├── tests/
│   ├── unit/                    # Component and isolated API/service tests
│   └── integration/             # Dataset and MLflow integration tests
├── requirements.txt
├── pytest.ini
└── README.md
```

The root-level `data/` directory holds datasets. References such as
`data/loader.py` in component handoffs are relative to
`src/predictive_maintenance/`, not the repository root.

## Data versioning with DVC

Git tracks source code, configuration, documentation, tests, and DVC metadata.
Dataset files must not be committed directly to Git.

Raw data is expected at:

```text
data/raw/predictive_maintenance_5000.csv
```

Use the original 5,000-row CSV, preserving its contents, row order, and original
column names. The training code creates the initial 3,000 training, 750
validation, 750 future, and 500 final-test partitions in memory with seed `42`.
Do not prepare separate split CSVs for the current initial-training workflow.

`data/raw.dvc` records the raw data version. The committed `.dvc/config` is
currently empty: it does not define a shared remote. Agree on remote access
with the team before relying on `dvc pull`. For current local development,
place the team's unchanged original CSV at the path above. It is Git-ignored.

## Running the API locally

Run all commands from the repository root. The CI target is Python 3.11.
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
http://127.0.0.1:5000. Run metadata is stored in `mlflow.db` and artifacts in
`mlartifacts/`. These generated files are Git-ignored. Keep the same database
and artifacts between restarts to retain access to logged models.

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

The server can start before `MODEL_URI` is configured. `/health` only checks
API responsiveness; `/predict` requires a configured model.

### Train and choose a prediction model

In Swagger UI, call `POST /train` with:

```json
{
  "dataset_id": "default",
  "model_type": "gradient_boosting",
  "random_state": 42
}
```

`xgboost` is also supported. Training is synchronous and returns a real
`training_run_id`, `model_uri`, and six validation metrics. Check the run and
logged model in the `predictive-maintenance-baselines` MLflow experiment.

Copy the returned `model_uri`. Stop the API with `Ctrl+C`, then configure it
in that same API terminal and restart. Replace the placeholder below with
the complete URI returned by your training run:

```powershell
$env:MODEL_URI = "models:/<model-id-from-train-response>"
python -m uvicorn predictive_maintenance.api.main:app --app-dir src --reload
```

The placeholder is not a usable model URI. Alternatively, use an existing
trusted logged model accessible through the configured MLflow backend.

Environment variables set this way apply to the current terminal and its
child processes. Set them again when starting from a new terminal.

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
probability. The response uses `mode: "real"`; prediction `1` means the
probability is at least `0.5`, otherwise the prediction is `0`.

`model_version` currently contains the logged-model identifier, such as
`m-...`; it is not a Model Registry version number.

The process caches the most recently loaded model/MLflow-URI pair.
Training does not automatically change the prediction model. To switch
models, update `MODEL_URI` and restart the API. Only configure trusted models.

## Available endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Check API responsiveness; returns `{"status": "ok"}`. |
| POST | `/train` | Train and evaluate a model, then log it to MLflow. |
| POST | `/predict` | Return a real prediction using the configured fitted pipeline. |

Successful requests return HTTP `200`. Invalid request bodies return `422`.
Handled operational failures return `503` or `500`; see the
[API contract](docs/api_contract.md) for exact fields and messages.

## Running tests and quality checks

The focused API and service tests do not require the real CSV, a configured
`MODEL_URI`, or a running MLflow server. They use controlled service/model
results and temporary synthetic data.

```powershell
python -m pytest tests/unit/test_api.py tests/unit/test_api_services.py -v
```

Run the complete test suite after making the original CSV available:

```powershell
python -m pytest -v
```

MLflow integration tests use isolated temporary storage. Dataset integration
tests require the real CSV; not all of them skip automatically when it is absent.

Check code style and types:

```powershell
python -m flake8 src tests infra app scripts
python -m pyright
```

The shared Pyright configuration references `.venv`. If that directory is
absent, Pyright may print an environment message. Resolve any actual errors
and make sure it uses the intended Python interpreter.

## Current scope and remaining work

Real training, MLflow model logging, model loading/cache, real prediction,
input validation, and basic HTTP error handling are implemented. Local
Swagger calls have demonstrated successful training, artifact loading,
and a real prediction. Automated tests verify the API and service behavior
with isolated data and controlled external calls; a passing unit test does
not by itself establish end-to-end MLflow availability.

Task 4.1 is not complete yet. Remaining work includes persistent prediction
request/result logging, retrievable prediction JSON/CSV artifacts in MLflow,
tests for those additions, and final repository-wide QA/coverage verification.
Model Registry activation, automated retraining, monitoring, and Docker
deployment belong to later integration/infrastructure work.

See the [API contract](docs/api_contract.md) for request/response details and
the [Git workflow](docs/branching.md) for feature branches, review, and merges.
