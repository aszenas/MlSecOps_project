# API Contract

This document describes verified DVC training with dataset versions `"v1"`
and `"v2"`, MLflow Registry champion selection, and prediction using the
current champion pipeline.

Base URL for local development: `http://127.0.0.1:8000`

Interactive documentation: `/docs`

OpenAPI specification: `/openapi.json`

POST endpoints accept JSON request bodies.

## GET /health

Checks whether the API is responding. It does not check dataset availability,
model availability, or external services.

### Successful response — 200 OK

```json
{
  "status": "ok"
}
```

## POST /predict

Accepts one machine-data record and returns a prediction calculated by the
current champion's fitted preprocessing and classifier pipeline.

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

The numeric fields use Pydantic's default type conversion rather than strict
type validation. For example, a numeric string such as `"300.8"` can be
converted to a number. Clients should send measurements as JSON numbers.

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

The following response is illustrative. Probabilities and concrete Registry
versions depend on the selected champion and input.

```json
{
  "prediction": 0,
  "probability": 0.1,
  "model_version": "2",
  "mode": "real"
}
```

| Field | Meaning |
| --- | --- |
| `prediction` | `0`: no failure; `1`: failure |
| `probability` | Probability of failure, between 0 and 1 |
| `model_version` | Concrete version of `predictive-maintenance-model` used for this prediction |
| `mode` | Always `"real"` for this endpoint |

### Prediction behavior and model configuration

The server resolves the `champion` alias of `predictive-maintenance-model`
before every prediction. It loads the immutable URI
`models:/predictive-maintenance-model/<version>`.

Clients cannot choose a model or Registry backend.
The old `MODEL_URI` environment variable is ignored.

`MLFLOW_TRACKING_URI` selects both the tracking and Registry backend.
Its default is `http://127.0.0.1:5000`.
An explicitly blank setting is rejected.

The service builds a single-row DataFrame using the six normalized feature
names in the model's expected order. The loaded pipeline performs preprocessing
internally; it is not fitted again during prediction.

The trained binary classifier uses class order `[0, 1]`.
The service reads the positive-class probability from `predict_proba`,
requires an output shape of `(1, 2)`, and rejects non-finite values or
probabilities outside `[0, 1]`.

The failure prediction is `1` when the failure probability is greater than
or equal to `0.5`; otherwise it is `0`.

The most recently loaded concrete `(model_uri, tracking_uri)` pair is cached
per process. Every request still checks the alias. The same version reuses
the pipeline; a promoted or rolled-back version is loaded on the next request.

The version is resolved once per request, so an alias change during processing
does not change that request's response or recorded model identity.

If alias lookup fails, the API does not silently serve a stale cached model.
Prediction does not load the CSV or refit preprocessing.

A successful `/train` may promote its candidate.
A rejected candidate leaves the champion unchanged.

### Operational errors

| Status | Detail |
| --- | --- |
| `503` | `No champion model is available.` |
| `503` | `The prediction model is unavailable.` |
| `500` | `Prediction failed because of invalid model data.` |
| `503` | `The prediction event could not be stored.` |

A missing model or champion alias returns the specific no-champion `503`.
Other handled MLflow, runtime, or operating-system failures return the
general unavailable-model `503`.

The `500` handler covers `ValueError`, including an incompatible loaded object
or invalid probability output.

These responses contain a string-valued `detail`, for example:

```json
{
  "detail": "The prediction model is unavailable."
}
```

Unexpected failures may also result in HTTP `500`.

### Persistent prediction events

Each request that passes input validation receives a server-generated UUID.
For successful predictions and handled prediction failures, the API commits
an event to a local SQLite database and returns the UUID in the `X-Request-ID`
response header. The JSON response schema is unchanged.

The event is stored in `prediction_events`, with `request_id` as the unique
primary key and `payload` containing JSON.

| Field | Meaning |
| --- | --- |
| `request_id` | UUID matching the response header |
| `timestamp` | Request processing start time in UTC ISO 8601 format |
| `inputs` | The six validated request features |
| `registered_model_name` | `predictive-maintenance-model` |
| `model_alias` | `models:/predictive-maintenance-model@champion` |
| `model_uri` | Concrete served version URI on success; null on a handled prediction failure |
| `decision_threshold` | `0.5` |
| `status` | `success` or `error` |
| `http_status` | Intended prediction status: `200`, `503`, or `500` |
| `prediction`, `probability`, `model_version`, `mode` | Successful prediction result fields |
| `error` | Public error message for a handled prediction failure |

The logged success URI is constructed from the version returned by prediction,
without resolving the alias a second time.

Failed events do not claim a served model version.
Historical records retain their original identifiers.
Internal exception messages and stack traces are not stored in these events.

Requests rejected with `422` do not reach this logging step.
Unhandled failures are not covered by this event logger.

`PREDICTION_LOG_DB` configures the SQLite file.
Its default is the repository's `artifacts/predictions.db`.
The parent directory and table are created on demand.

Committed events remain available when the API restarts with the same file.
The generated database must not be committed to Git.
Docker Compose places this database on the persistent `prediction_data` volume.

If the event cannot be committed, the API returns `503` with
`The prediction event could not be stored.` and an `X-Request-ID` header.

This also takes precedence if persistence fails while recording a handled
prediction error. No event can be guaranteed in a storage-failure case.

There is no automatic retry or retention policy.
A separate command exports persisted events to MLflow.

## Prediction artifact export (command-line operation)

`python scripts/export_predictions.py` exports a snapshot of persisted events
to the `predictive-maintenance-predictions` MLflow experiment.

Run from the repository root with `PYTHONPATH=src`.
This is not an HTTP endpoint and is not automatically executed by `/predict`.

| Option | Behavior |
| --- | --- |
| `--database PATH` | Override `PREDICTION_LOG_DB` / the default `artifacts/predictions.db` |
| `--tracking-uri URI` | Override `MLFLOW_TRACKING_URI` / the default `http://127.0.0.1:5000` |
| `--all` | Create a historical snapshot, including already uploaded events |
| `--model-uri URI` | Select the exact stored `model_uri` |
| `--since TIMESTAMP` | Inclusive event timestamp boundary, with timezone |
| `--until TIMESTAMP` | Exclusive event timestamp boundary, with timezone |

Default mode selects events not included in a completed incremental batch
for the same tracking URI string. Both success and error events are eligible.

Historical snapshots never remove events or alter incremental eligibility.
No matching events means no remote run and output
`{"status": "no_matching_events"}`.

Each non-empty batch stores an immutable JSON snapshot and unique `batch_id`
in SQLite before contacting MLflow.

Its artifact path is `predictions/predictions.json`.
The document contains `schema_version`, `batch_id`, `created_at`,
`selection`, `event_count`, and an `events` array.
The original event fields are preserved, including `request_id`.

Successful output includes `run_id` and a downloadable URI in the form
`runs:/<run-id>/predictions/predictions.json`.

The run has `operation=predict` and `batch_id` tags, an event count,
model URI information, and an export mode.
Any previous artifact remains independently downloadable from MLflow UI.

Incremental bookkeeping is completed only after artifact upload and remote
run completion. Failures leave the local batch pending.

Retrying the same selection reuses its snapshot, batch ID, remote run,
and artifact path.

Remote runs can also be recovered by batch tag if creation succeeded before
the run ID was saved locally. A finished remote run allows local recovery
without repeating its upload.

This is recovery coordination between two stores, not a distributed atomic
transaction.

An unfinished selection must be retried before changing filters for the same
destination. Multiple remote runs with the same batch tag are reported as an
error rather than silently selected.

Exporters are serialized using a separate local SQLite lock file;
event writes remain possible during upload.

CLI errors produce a nonzero exit status.
No scheduler, pruning, or deletion is performed.

See the README for command examples and restart/retry instructions.

## POST /train

Verifies the selected Git/DVC training data, trains and logs a candidate,
registers a model version, and completes audited champion selection.

### Request fields

All request fields have defaults.
Additional fields and explicit null values are rejected.

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `dataset_id` | string | `"default"` | Only `"default"` is accepted |
| `dataset_version` | string | `"v1"` | `"v1"` or `"v2"` |
| `model_type` | string | `"gradient_boosting"` | `"gradient_boosting"` or `"xgboost"` |
| `random_state` | integer | `42` | Only `42` is accepted |

`dataset_id` identifies the configured dataset.
Both versions require the original raw file:

```text
data/raw/predictive_maintenance_5000.csv
```

Version `"v2"` additionally requires:

```text
data/processed/training/train_v2.csv
```

These paths are relative to the server-controlled `TRAINING_REPOSITORY_ROOT`,
which defaults to the application repository root.

An override must be an absolute path; explicitly blank values are rejected.
It refers to dataset storage, not Python modules under
`src/predictive_maintenance/data/`.

Clients cannot supply an arbitrary file path.

`model_type` selects either Gradient Boosting or XGBoost.

`random_state` is restricted to the numeric literal `42`, matching the fixed
splitting and model-training configuration. The string `"42"` is not accepted.

### Data preparation and training

The original raw dataset must contain exactly 5,000 records with the
original CSV columns.

The training service:

1. Verifies a clean Git checkout and unchanged raw DVC dataset.
2. Loads and validates the raw CSV, normalizes columns, and separates features/target.
3. Creates the fixed initial partitions.
4. For `"v2"`, verifies and loads the expanded training CSV against the expected split.
5. Rechecks Git/DVC references before training.
6. Trains, evaluates and logs the selected candidate.
7. Registers the model and logs dataset lineage and the exact split manifest.
8. Compares candidate/champion on validation data and records the decision.
9. Promotes an eligible candidate and verifies the alias, or keeps the champion.

The initial `"v1"` partitions are:

| Partition | Records | Purpose |
| --- | ---: | --- |
| Training | 3,000 | Fit preprocessing and the classifier |
| Validation | 750 | Calculate validation metrics |
| Future | 750 | Reserved rows available for monitoring and controlled retraining |
| Final test | 500 | Reserved for final evaluation |

Training uses only the training partition.
The operation calculates metrics on the validation partition;
it does not evaluate the final-test partition.

### Training dataset versions

`dataset_version` selects the training data version.
Omitting it uses `"v1"`.

- `"v1"` uses the initial training partition.
- `"v2"` loads the DVC-verified `data/processed/training/train_v2.csv`
  and verifies that it matches the expected expanded training partition.

Version `"v2"` moves predefined labeled future rows into training.
Those rows are removed from the future partition.
The original validation and final-test partitions remain unchanged.

Both versions require the verified original raw dataset.
The partition sizes in the table above describe `"v1"`.

Unsupported versions and explicit null values return HTTP `422`
before training starts.

Dataset versions (`"v1"` and `"v2"`) are separate from numeric MLflow
Registry model versions.

The response schema is unchanged; `dataset_version` is a request field.
A completed `"v2"` training run does not guarantee champion promotion.

### Example requests

Initial training:

```json
{
  "dataset_id": "default",
  "dataset_version": "v1",
  "model_type": "gradient_boosting",
  "random_state": 42
}
```

Training with the expanded labeled dataset:

```json
{
  "dataset_id": "default",
  "dataset_version": "v2",
  "model_type": "gradient_boosting",
  "random_state": 42
}
```

An empty JSON object is also accepted and uses all defaults, including `"v1"`:

```json
{}
```

The request body itself is required.
Omitting it is different from sending `{}` and returns HTTP `422`.

### Successful response — 200 OK

The following response is illustrative.
Identifiers and metric values depend on the actual training run.

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
  "mode": "real",
  "registered_model_name": "predictive-maintenance-model",
  "registered_version": "2",
  "champion_version": "2",
  "champion_version_before": "1",
  "promoted": true,
  "selection_reason": "higher_f1"
}
```

| Field | Meaning |
| --- | --- |
| `status` | `"completed"` means training and selection completed, including a rejected candidate |
| `training_run_id` | The actual MLflow training run ID |
| `model_type` | The selected classifier |
| `model_uri` | URI used to load the logged preprocessing and model pipeline |
| `metrics` | Six candidate metrics calculated on the validation partition |
| `registered_model_name` | Registered model name |
| `registered_version` | Newly registered candidate version |
| `champion_version` | Selected champion version, or null when no candidate has qualified |
| `champion_version_before` | Champion before this selection, or null |
| `promoted` | Whether this operation promoted the candidate |
| `selection_reason` | Decision reason, such as `higher_f1`, `first_champion`, `f1_not_improved`, or `recall_below_minimum` |
| `mode` | Always `"real"` for this endpoint |

Training is synchronous: the response is returned after the operation finishes.
No background job is created.

The operation logs the following to MLflow:

- Model type, training parameters, and dataset partition sizes.
- Validation metrics.
- Git commit and dataset lineage.
- The exact split manifest.
- The complete fitted preprocessing and classifier pipeline.
- The Registry selection decision.

For `"v2"`, dataset lineage also includes the verified training-file DVC hash.

The returned `model_uri` identifies the logged candidate artifact,
which may have been rejected.
Use `champion_version` to identify the selected serving model.

Promotion requires recall >= 0.70 and, when a champion exists,
a strictly higher validation F1.
An equal F1 or insufficient recall preserves the champion.

With no previous champion, a rejected first candidate returns
`champion_version: null` and `promoted: false`;
`/predict` then returns `503`.

On successful promotion, the next prediction uses the new version
without an API restart.
Rejected versions remain registered and linked to their run.

Concurrent selection/promotion coordination remains a separate limitation
of the underlying Registry workflow.
Do not run multiple promotion jobs at once.

### Operational requirements

The operation requires:

- A clean Git training checkout.
- Valid `data/raw.dvc` and the restored unchanged raw CSV.
- For `"v2"`, valid `data/processed/training/train_v2.csv.dvc`
  and the restored unchanged training CSV.
- DVC available in the API Python environment.
- The project's Python dependencies.
- An accessible MLflow tracking/Registry backend.
- Registration/alias permissions and writable artifact storage.

For local training, restore the required data from the repository root:

```powershell
python -m dvc pull data/raw.dvc data/processed/training/train_v2.csv.dvc -r dagshub
```

Docker Compose performs these downloads and checks in its preparation service.

The training code uses `MLFLOW_TRACKING_URI` when set.
Otherwise, it defaults to:

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
| `503` | `The training data or registry service is unavailable.` |
| `500` | `Training failed because of invalid data or configuration.` |

These responses contain a string-valued `detail` field:

```json
{
  "detail": "The training dataset is unavailable."
}
```

A `FileNotFoundError` produces the dataset-unavailable `503`.
Git/DVC verification runtime failures and operating-system failures use
the training-data/Registry `503`.

Validation errors raised by provenance or training use the
invalid-data/configuration `500`.
This includes the current provenance check for a missing or mismatched
`"v2"` training file when it raises `ValueError`.

Internal diagnostic details are not returned.
Unexpected failures may also result in HTTP `500`.

## Validation errors — 422 Unprocessable Entity

Invalid request bodies return HTTP `422`.

Examples include:

- Missing required prediction fields.
- A missing POST request body.
- Unsupported product types, dataset identifiers, dataset versions, or model types.
- Values outside the allowed ranges.
- Values that cannot be converted to the required numeric types.
- A `random_state` value other than the numeric literal `42`.
- Additional fields not defined by the request schema.
- Explicit null values.

The response contains a `detail` list with information about each validation error.

| Property | Meaning |
| --- | --- |
| `loc` | Location of the invalid value, such as `["body", "dataset_version"]` |
| `type` | Validation error identifier |
| `msg` | Human-readable explanation |

Additional properties may be included.
Clients should not depend on the exact wording of `msg`.

Request validation takes place before the endpoint function runs.
Invalid training requests do not start training;
invalid prediction requests do not invoke the prediction service.

## Current scope and remaining integration

`/train` delegates to verified DVC training and audited Registry selection.
It supports dataset versions `"v1"` and `"v2"`, defaulting to `"v1"`.

`/predict` resolves the champion per request, caches concrete versions and
records the actually served version.
Persistent prediction storage and JSON artifact export are implemented.

The retained mock services are not used by current routes.
`/health` only checks API responsiveness.

Docker Compose provides the API, MLflow, persistent storage and a temporary
Streamlit page.
The preparation service restores and verifies the raw data and `"v2"` training file.

API and service unit tests isolate external dependencies.
Dedicated integration tests use real temporary MLflow model artifacts
and Registry aliases to verify promotion, rejection, backend selection,
cache updates and event identities.

Their DVC training adapter is controlled;
real-data DVC training is covered separately by lineage integration tests.

The API-log monitoring module is available.
Connecting its runner, scheduled monitoring, drift-triggered retraining
and the final dashboard remains subsequent integration work.

Passing focused tests does not establish full-project CI or deployment status.

Any changes to the public API contract must also be reflected in the
schemas, examples, tests, and this document.
