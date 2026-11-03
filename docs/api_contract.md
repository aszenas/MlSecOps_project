# API Contract

This document describes the current API: real model training with MLflow tracking and mock prediction.

Base URL for local development: `http://127.0.0.1:8000`

Interactive documentation: `/docs`  
OpenAPI specification: `/openapi.json`

POST endpoints accept JSON request bodies.

## GET /health

Checks whether the API is responding. It does not check dataset availability, model availability, or external services.

### Successful response — 200 OK

```json
{
  "status": "ok"
}
```

## POST /predict

Accepts one machine-data record and returns a fixed mock prediction.

### Request fields

All fields are required. Null values and additional fields are rejected.

| Field | Type | Constraints | Meaning |
| --- | --- | --- | --- |
| `type` | string | `"L"`, `"M"` or `"H"` | Product type |
| `air_temperature_k` | number | Finite, greater than 0 | Air temperature in kelvin |
| `process_temperature_k` | number | Finite, greater than 0 | Process temperature in kelvin |
| `rotational_speed_rpm` | number | Finite, at least 0 | Rotational speed in revolutions per minute |
| `torque_nm` | number | Finite, at least 0 | Torque in newton-metres |
| `tool_wear_min` | number | Finite, at least 0 | Tool usage time in minutes |

The numeric fields use Pydantic's default type conversion rather than strict type validation. For example, a numeric string such as `"300.8"` can be converted to a number. Clients should send measurements as JSON numbers.

### Example request

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

### Successful response — 200 OK

```json
{
  "prediction": 0,
  "probability": 0.1,
  "model_version": "mock-v1",
  "mode": "mock"
}
```

| Field | Meaning |
| --- | --- |
| `prediction` | `0`: no failure; `1`: failure |
| `probability` | Probability of failure, between 0 and 1 |
| `model_version` | Model identifier; currently `"mock-v1"` |
| `mode` | Currently always `"mock"` for this endpoint |

Every valid request currently returns the response shown above. The input is validated but is not used to calculate a prediction.

The returned probability is a fixed test value, not a real model estimate.

## POST /train

Loads the configured initial dataset, trains the selected model, evaluates it on the validation partition, and logs the results to MLflow.

### Request fields

All request fields have defaults. Additional fields and explicit null values are rejected.

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `dataset_id` | string | `"default"` | Only `"default"` is accepted |
| `model_type` | string | `"gradient_boosting"` | `"gradient_boosting"` or `"xgboost"` |
| `random_state` | integer | `42` | Only `42` is accepted |

`dataset_id` identifies the configured dataset:

```text
data/raw/predictive_maintenance_5000.csv
```

This path is relative to the repository root. It refers to the root-level data directory, not the Python modules under `src/predictive_maintenance/data/`.

Clients cannot supply an arbitrary file path.

`model_type` selects either Gradient Boosting or XGBoost.

`random_state` is restricted to the numeric literal `42`, matching the fixed initial splitting and model-training configuration. The string `"42"` is not accepted.

### Data preparation and training

The initial dataset must contain exactly 5,000 records with the original raw CSV columns.

The training service:

1. Loads and validates the raw CSV.
2. Normalizes column names.
3. Separates the six input features from the binary target.
4. Creates reproducible, stratified dataset partitions.
5. Trains and evaluates the selected model.
6. Logs the training results to MLflow.

| Partition | Records | Purpose |
| --- | ---: | --- |
| Training | 3,000 | Fit preprocessing and the classifier |
| Validation | 750 | Calculate validation metrics |
| Future | 750 | Reserved for later monitoring and retraining work |
| Final test | 500 | Reserved for final evaluation |

Training uses only the training partition. The current training operation calculates metrics on the validation partition; it does not evaluate the final-test partition.

This splitting procedure is intended for the initial dataset. Future retraining must preserve the fixed validation and final-test partitions.

### Example request

```json
{
  "dataset_id": "default",
  "model_type": "gradient_boosting",
  "random_state": 42
}
```

An empty JSON object is also accepted and uses the defaults:

```json
{}
```

The request body itself is required. Omitting it is different from sending `{}` and returns HTTP `422`.

### Successful response — 200 OK

The following response is illustrative. The identifiers and metric values depend on the actual training run.

```json
{
  "status": "completed",
  "training_run_id": "<actual-run-id>",
  "model_type": "gradient_boosting",
  "model_uri": "<actual-model-uri>",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.80,
    "recall": 0.75,
    "f1": 0.77,
    "roc_auc": 0.90,
    "average_precision": 0.85
  },
  "mode": "real"
}
```

| Field | Meaning |
| --- | --- |
| `status` | `"completed"` indicates successful completion |
| `training_run_id` | The actual MLflow training run ID |
| `model_type` | The selected classifier |
| `model_uri` | URI used to load the logged preprocessing and model pipeline |
| `metrics` | Six metrics calculated on the validation partition |
| `mode` | Always `"real"` for this endpoint |

Training is synchronous: the response is returned after the operation finishes. No background job is created.

The operation logs the following to MLflow:

- Model type, training parameters, and dataset partition sizes.
- Validation metrics.
- Git commit and experiment metadata.
- The complete fitted preprocessing and classifier pipeline.

The returned `model_uri` identifies a logged model. It does not by itself indicate that a Model Registry version or champion alias has been created.

Training does not automatically activate the new model for prediction. The `/predict` endpoint currently remains mock-based.

### Operational requirements

The operation requires:

- The configured CSV file.
- The project's Python dependencies.
- An accessible MLflow tracking backend and writable artifact storage.

The training code uses the `MLFLOW_TRACKING_URI` environment variable when set. Otherwise, it defaults to:

```text
http://127.0.0.1:5000
```

The request cannot override the MLflow tracking URI.

### Operational errors

The route explicitly handles the following failures:

| Status | Detail |
| --- | --- |
| `503` | `The training dataset is unavailable.` |
| `503` | `The MLflow operation could not be completed.` |
| `500` | `Training failed because of invalid data or configuration.` |

These responses contain a string-valued `detail` field:

```json
{
  "detail": "The training dataset is unavailable."
}
```

Unexpected failures may also result in HTTP `500`.

## Validation errors — 422 Unprocessable Entity

Invalid request bodies return HTTP `422`.

Examples include:

- Missing required prediction fields.
- A missing POST request body.
- Unsupported product types, dataset identifiers, or model types.
- Values outside the allowed ranges.
- Values that cannot be converted to the required numeric types.
- A `random_state` value other than the numeric literal `42`.
- Additional fields not defined by the request schema.
- Explicit null values.

The response contains a `detail` list with information about each validation error.

| Property | Meaning |
| --- | --- |
| `loc` | Location of the invalid value, such as `["body", "torque_nm"]` |
| `type` | Validation error identifier |
| `msg` | Human-readable explanation |

Additional properties may be included. Clients should not depend on the exact wording of `msg`.

Request validation takes place before the endpoint function runs. Invalid training requests do not start training.

## Current scope and remaining integration

The `/train` endpoint is connected to the real training pipeline.

The `/predict` endpoint still returns fixed mock results. The following parts are not implemented yet:

- Loading a model for real API predictions.
- Returning predictions and probabilities calculated by that model.
- Persistent prediction-request and result logging.
- Prediction JSON/CSV artifact logging to MLflow.
- Automatic model activation or Registry-based model selection.

The `/health` endpoint checks API responsiveness only.

API unit tests replace the training service with controlled test results. Passing those tests does not by itself demonstrate successful training against a running MLflow backend. Actual training and artifact retrieval require a separate integration check.

Any changes to the public API contract must also be reflected in the schemas, examples, tests, and this document.