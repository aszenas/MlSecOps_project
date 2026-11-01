# MlSecOps_project
MlSecOps beadando @beni and @anna

```text
MlSecOps_project/
├── app/                         # Streamlit dashboard
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Preprocessed datasets
├── docs/                        # Project documentation
├── infra/                       # Docker and infrastructure configuration
├── src/
│   └── predictive_maintenance/
│       ├── api/                 # REST API endpoints and schemas
│       ├── data/                # Data loading and preprocessing
│       ├── models/              # ML training, prediction and MLflow
│       └── monitoring/          # Drift detection, logging and retraining
├── tests/                       # Unit and integration tests
└── README.md                    # Project overview
```

## Running the API locally

Run the following commands from the project root.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Start the development server

```powershell
python -m uvicorn predictive_maintenance.api.main:app --app-dir src --reload
```

The `--app-dir src` option makes the application package available for import.
The `--reload` option restarts the development server when source files change.

Once the server is running:

- API base URL: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI specification: http://127.0.0.1:8000/openapi.json

Stop the server with `Ctrl+C`.

## Available endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Check that the API is responding. |
| POST | `/predict` | Validate machine data and return a fixed mock prediction. |
| POST | `/train` | Validate training settings and return a simulated completion result. |

Use **Try it out** in Swagger UI to send the provided example requests.

Successful requests return HTTP `200`.
Invalid request bodies return HTTP `422` with validation details.

## Current mock behavior

The API currently runs without a dataset, trained model, or MLflow service.

- `/health` returns `status: "ok"` and `mode: "mock"`.
- `/predict` always returns prediction `0`, failure probability `0.1`,
  model version `"mock-v1"`, and mode `"mock"` for valid input.
- `/train` returns status `"mock_completed"`, a null training run ID,
  and mode `"mock"`.
- Training does not create or update a model.
- Prediction values are simulated and must not be interpreted as real model results.

## Running tests

```powershell
python -m pytest -v
```

The tests cover the health endpoint, successful mock prediction and training
requests, and invalid request bodies.

A running Uvicorn server is not required: the tests use FastAPI's `TestClient`
to call the application directly.

For request fields, response formats, defaults, and validation rules,
see the [API contract](docs/api_contract.md).