# MLSecOps — Model Training & MLflow Tracking (3.1)

**Project:** Predictive Maintenance / MLSecOps  
**Development branch:** `feature/mlflow-training` (based on `dev`)  
**Documentation date:** 1 November 2026  
**Purpose:** Describe the implemented 3.1 training and MLflow work, its quality checks and limitations, and the exact integration contract for Máté's **4.1/10 — Train/predict wiring and API tests** task.

> **Status convention:** **Implemented** means code has been developed and exercised in the conversation; **verified locally** means a reported run/test passed; **pending** means not included in 3.1. This document does not claim a successful remote GitHub Actions run, PR approval/merge, deployed API, MLflow Registry integration, or prediction-request logging.

## 1. Goals and scope

The previous 2.1 preprocessing layer is reused to train binary machine-failure classifiers on the AI4I predictive-maintenance dataset. Step 3.1 introduces reusable model training, shared validation metrics, MLflow experiment tracking, model persistence and loading, reproducibility verification, and automated tests. Training logic belongs under `src/predictive_maintenance/models/`; scripts provide command-line entry points; FastAPI routes remain the API developer's responsibility.

**Implemented:** Gradient Boosting and XGBoost baseline models; one fitted sklearn Pipeline combining preprocessing and classifier; train-only balanced sample weights; six validation metrics; MLflow experiment/run logging; complete model logging in `cloudpickle` format; model loading and independent reproduction checks; mock-based unit tests and isolated SQLite-backed MLflow integration tests.

**Not implemented by 3.1:** HTTP `/train` or `/predict` wiring, request/response schema validation at the API boundary, asynchronous job management, prediction-level persistent audit logging, prediction JSON/CSV artifact logging, Model Registry stage/alias selection, remote/shared MLflow provisioning, hyperparameter tuning, threshold optimization, drift detection and production deployment.

## 2. Dataset, leakage prevention and split policy

- Data file used for local real-data experiments: `data/raw/predictive_maintenance_5000.csv` (5,000 rows). It is locally available but may be absent from CI until the separate DVC work provides it.
- Binary target: `target`, where `1` represents machine failure. The original dataset contains 163 positive cases out of 5,000 (~3.26%).
- Model inputs: `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min` (**six raw features**). The preprocessing Pipeline generates `temperature_difference_k`; it then imputes, scales and one-hot encodes to nine features on the current data.
- `udi`, `product_id`, `failure_type`, and the label `target` **must never enter inference features**; in particular, `failure_type` leaks the outcome.
- Stratified, fixed `random_state=42` initial split: **3,000 train / 750 validation / 750 simulated future / 500 final test**.
- Fit the entire model Pipeline **only** using training inputs and labels. Validation is for comparing models and selecting future thresholds. Future is reserved for drift/retraining simulation, and final test remains untouched until final assessment.
- Recalculate `calculate_sample_weights(y_train)` when a new training set is used. Do not combine the chosen balanced `sample_weight` with a second imbalance multiplier such as XGBoost `scale_pos_weight` calculated from the same ratio.

## 3. Architecture and file map

```text
src/predictive_maintenance/
├── data/
│   ├── loader.py                 # load_raw_data: strict original CSV validation
│   ├── preprocessing.py          # DataProcessor and fitted preprocessing stages
│   ├── splitting.py              # DataSplits and create_initial_data_splits
│   ├── imbalance.py              # calculate_sample_weights(y_train)
│   └── persistence.py            # optional trusted standalone preprocessing joblib
├── models/
│   ├── training.py               # build_model_pipeline, train_model, ModelType
│   ├── evaluation.py             # evaluate_model (six metrics)
│   └── mlflow_tracking.py        # run_training_experiment, EXPERIMENT_NAME
└── api/
    ├── main.py                   # existing API skeleton (owned by API developer)
    ├── routes.py                 # endpoint definitions (API developer)
    ├── schemas.py                # API request/response schemas (API developer)
    └── mock_service.py           # skeleton/mock service (to replace/wire)

scripts/
├── run_baseline_experiments.py   # local entry point; trains and logs both models
└── verify_mlflow_model.py        # local CLI: reload and verify saved model

tests/
├── unit/
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_mlflow_tracking.py   # mock-based tracking tests
│   └── test_api.py               # existing API-skeleton tests
└── integration/
    ├── test_model_training.py    # dataset-dependent real-data model test
    └── test_mlflow_integration.py # synthetic data + temporary SQLite MLflow

docs/
├── mlflow_training_branch_documentation_en.md
└── mlflow_training_branch_documentation_hu.md
.github/workflows/python-qa.yml   # CI gates to be reviewed/pushed
requirements.txt                 # MUST include mlflow, xgboost and API dependencies
```

The listed paths reflect the development structure discussed and implemented. The exact on-disk repository state and the current API skeleton must be reviewed on the integration branch before merge.

## 4. Model training implementation

### 4.1 `models/training.py`

**Public interface:**

```python
ModelType = Literal["gradient_boosting", "xgboost"]

build_model_pipeline(preprocessor, model_type="gradient_boosting") -> Pipeline
train_model(X_train, y_train, preprocessor, model_type="gradient_boosting") -> Pipeline
```

- `build_model_pipeline` clones the **unfitted** preprocessor and adds `GradientBoostingClassifier(random_state=42)` or `XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=1.0, n_jobs=1)`.
- `train_model` rejects empty feature/target inputs, mismatched lengths, missing targets, and targets lacking either class; it obtains balanced weights from `data/imbalance.py` and fits using `model__sample_weight=weights.to_numpy()`.
- The return value is one **fitted** sklearn Pipeline containing both `preprocessor` and `model` steps. It accepts the original six **normalized** feature columns at prediction time and exposes `predict(X)` and `predict_proba(X)`.
- It is a synchronous operation. It does not create an MLflow run unless wrapped by the tracking function below.

### 4.2 `models/evaluation.py`

```python
evaluate_model(model, X_validation, y_validation, threshold=0.5) -> dict[str, float]
```

Returns exactly `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, and `average_precision`. It uses the positive-class probability (`predict_proba(X)[:, 1]`) and validates binary targets, input lengths, probability shape/finiteness and a decision threshold in `(0,1)`. Precision/recall/F1 use `zero_division=0` (with a local typing workaround where required). No final-test evaluation is performed by this function automatically; callers choose the intended partition.

### 4.3 `models/mlflow_tracking.py`

```python
EXPERIMENT_NAME = "predictive-maintenance-baselines"

run_training_experiment(
    splits: DataSplits,
    model_type: ModelType,
    tracking_uri: str | None = None,
) -> dict[str, object]
```

The routine selects the tracking URI (argument, then `MLFLOW_TRACKING_URI`, then `http://127.0.0.1:5000`), creates/selects the experiment, starts a named run, constructs a fresh `DataProcessor().create_preprocessing_pipeline()`, calls `train_model`, evaluates on **validation** only, and logs:

- Run name, model type, decision threshold (`0.5`), `random_state=42`, balanced weighting and available scalar classifier hyperparameters.
- Dataset partition counts (`train_size`, `validation_size`, `future_size`, `test_size`).
- `validation_accuracy`, `validation_precision`, `validation_recall`, `validation_f1`, `validation_roc_auc`, `validation_average_precision`.
- Tags `project`, `stage`, `evaluation_split`, and Git commit SHA (or `unknown` when unavailable).
- **Full fitted Pipeline** with `mlflow_sklearn.log_model(sk_model=pipeline, name="model", input_example=..., serialization_format="cloudpickle", code_paths=["src"])`.

Return keys are `run_id`, `model_type`, `model_uri` and `metrics`. The `model_uri` is an MLflow 3 logged-model URI such as `models:/m-...` and is **not necessarily a registered model name/version**. No prediction input/output records are logged here.

**Important API integration detail:** the current function **trains and evaluates synchronously from a complete `DataSplits` object**, not from an arbitrary single-row JSON request. Máté must decide where training data is loaded and how asynchronous scheduling/timeouts are handled at the HTTP layer.

## 5. Local execution and recorded baseline results

Create an activated Python 3.11 virtual environment and install `requirements.txt`. Start a local tracking server (when needed) with:

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --workers 1
```

The browser UI is `http://127.0.0.1:5000`, **Model training** → `predictive-maintenance-baselines`. This local service does not require an MLflow Cloud account. Keep it bound to localhost; exposing a server on a shared network requires appropriate access controls.

Run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_baseline_experiments.py
```

On **1 November 2026**, the user reported these **local validation** results (threshold 0.5; 750 records):

| Metric | Gradient Boosting | XGBoost |
|---|---:|---:|
| Accuracy | 0.9720 | 0.9827 |
| Precision | 0.5484 | 0.7391 |
| Recall | 0.7083 | 0.7083 |
| F1 | 0.6182 | 0.7234 |
| ROC-AUC | 0.9716 | 0.9706 |
| Average precision | 0.7568 | 0.7867 |

XGBoost has higher F1 and precision with equal recall on this validation split; no claim of universal superiority or final-test performance is made. Two successful MLflow runs produced logged model identifiers:

- Gradient Boosting: `models:/m-e512544fa021420ea31cd9e42d529826`
- XGBoost: `models:/m-fee7392716644fcfa13a3dce16c776d6`

**These identifiers point to the original local tracking server**, not to a teammate's fresh MLflow instance. Transfer/retraining/shared storage is required before other machines can resolve them. They are examples, not hard-coded production defaults.

## 6. Verification and quality assurance

### 6.1 Manual reproducibility verification

`scripts/verify_mlflow_model.py` accepts `--model-uri` and `--model-type`; it rebuilds the original split and trains a reference model using the current code/settings, then loads the MLflow model and compares class predictions, probabilities (`rtol=atol=1e-10`) and validation metrics. The user provided an explicit **PASS** output for Gradient Boosting on 750 records; XGBoost was also described as passing, but its separate terminal output was not provided in the conversation. This check verifies the baseline reproducibility **under matching data/code/dependency versions**. It is not a generalized verifier for future tuned models.

```powershell
$env:PYTHONPATH = "src"
python scripts/verify_mlflow_model.py --model-uri "models:/m-e512544fa021420ea31cd9e42d529826" --model-type gradient_boosting
python scripts/verify_mlflow_model.py --model-uri "models:/m-fee7392716644fcfa13a3dce16c776d6" --model-type xgboost
```

### 6.2 Automated testing

- `tests/unit/test_training.py`: model selection, prediction shape, repeatability, invalid training inputs.
- `tests/unit/test_evaluation.py`: expected metrics, custom threshold, input and probability validation.
- `tests/unit/test_mlflow_tracking.py`: mock-based run parameters, metrics, model-artifact call, tracking URI choice and Git metadata.
- `tests/integration/test_model_training.py`: AI4I CSV-based fit and inference; requires dataset availability.
- `tests/integration/test_mlflow_integration.py`: **real training and MLflow artifact round trip** for both model types using synthetic feature-compatible data and a **temporary SQLite backend**. No HTTP server or AI4I CSV required. Its URI must use the SQLite dialect, e.g. `f"sqlite:///{(tmp_path / 'mlflow_test.db').as_posix()}"`; MLflow 3.17.2 rejects the legacy file-store backend by default.

Reported local `src/` coverage after unit and MLflow integration tests: **314 / 316 statements = 99.37%**. Breakdown: `training.py` 100%, `mlflow_tracking.py` 100%, `evaluation.py` 95.65%; missed lines were `data/loader.py:96` and `models/evaluation.py:54`. These figures are local run results, **not CI certification**.

```powershell
$env:PYTHONPATH = "src"
python -m coverage erase
python -m coverage run -m pytest tests/unit/ -v
python -m coverage run --append -m pytest tests/integration/test_mlflow_integration.py -v
python -m coverage report -m --include="src/*" --fail-under=95
python -m flake8 src tests infra app scripts
python -m pyright
```

Static lint/type checks must cover `scripts/`, but `scripts/` is **excluded from the mandatory 95% coverage target**, which is measured for `src/` only. The CSV-dependent integration suite runs separately once data is available.

### 6.3 GitHub Actions acceptance criteria (verify before merge)

In `.github/workflows/python-qa.yml`: install `requirements.txt` and QA tools, run unit tests, run MLflow SQLite integration tests **unconditionally**, run CSV-dependent integration tests only when the dataset is present (excluding `test_mlflow_integration.py` to avoid double execution), check `src/` coverage >=95%, run Flake8 on `src tests infra app scripts`, and run Pyright including `scripts/`. Record an explicit warning if CSV-based tests are skipped until DVC is integrated.

The repo's `requirements.txt` snapshot supplied during development listed `mlflow==3.17.2` and the FastAPI dependencies but **did not list XGBoost**. Add `xgboost` at the tested version before CI; verify a clean Python 3.11 installation. No successful CI run or final merge is asserted in this document.

## 7. Operational and security limitations

1. `cloudpickle` is used because the Pipeline contains a custom feature engineering function and tree-based models. **Only load trusted artifacts**: pickle-compatible formats may execute arbitrary code during deserialization. Do not load arbitrary user-provided model URIs.
2. The MLflow signature inference warned about integer columns with potential missing values. Normalize inference dtypes/nullable input and explicitly test schema behavior before enabling production `/predict`.
3. Local model IDs are scoped to the tracking backend; do not hard-code local URIs in shared API deployment. Registry integration and serving model selection remain separate tasks.
4. Prediction request logging is **not** part of the existing training tracking routine. Logging real requests should minimize or redact identifying/sensitive data, enforce retention/access controls and avoid storing secrets.
5. With MLflow 3.17.2, the filesystem tracking backend is in maintenance mode. Use SQLite for isolated tests and a deliberate database/artifact-store configuration for shared environments.
6. The `/train` call is synchronous; HTTP timeouts, concurrency, job identifiers and long-running job scheduling belong to API orchestration.
7. Baseline threshold is fixed at 0.5. Threshold optimization, automated retraining, final-test reporting, drift detection and production monitoring are not claimed.

---

# 8. Handoff to Máté — Task 4.1/10: functional `/train` and `/predict`

**Máté's assignment:** Connect the existing API skeleton to real preprocessing and ML models; expose working `/train` and `/predict` endpoints with input validation, JSON responses, model loading, error handling, Swagger/Postman usability, tests, MLflow train-run logs, and **persistent prediction request/result metadata plus retrievable prediction JSON/CSV artifacts in MLflow**. The API layer must not duplicate the already implemented preprocessing or model-training code.

## 8.1 Exact reuse map

| Máté needs | Existing location / symbol | How to use it | Still to implement in API |
|---|---|---|---|
| Read and validate original training CSV | `data/loader.py` → `load_raw_data(path)` | Apply strict raw 10-column CSV validation | Choose dataset source/path; useful 4xx/5xx errors |
| Normalize training columns | `data/preprocessing.py` → `DataProcessor.rename_columns(df)` | Raw headers → snake_case | Control entrypoint/data source |
| Separate features and labels | `DataProcessor.split_features_target(df)` | Returns `(X, y)` with only 6 selected features | Ensure no label leakage |
| Initial fixed partition | `data/splitting.py` → `create_initial_data_splits(X,y)` → `DataSplits` | 3000/750/750/500 (only for initial 5000-row dataset) | Decide train data selection and future retraining semantics |
| Create unfitted preprocessing | `DataProcessor.create_preprocessing_pipeline()` | Feature engineering, imputation, scaling, one-hot encoding | Do not fit this on `/predict` requests |
| Balanced training weights | `data/imbalance.py` → `calculate_sample_weights(y_train)` | Already invoked in `train_model` | Do not duplicate |
| Train fitted model Pipeline | `models/training.py` → `train_model(X_train,y_train,preprocessor,model_type)` | Return fitted Pipeline | HTTP orchestration if bypassing MLflow wrapper |
| Validate model quality | `models/evaluation.py` → `evaluate_model(model,X_validation,y_validation)` | Six metrics at threshold 0.5 | Format and return relevant fields |
| Train **and** log to MLflow | `models/mlflow_tracking.py` → `run_training_experiment(splits,model_type,tracking_uri=None)` | Preferred `/train` integration point; logs parameters, six metrics, full fitted model | Handle input, scheduling, errors, response, remote tracking URI |
| Run two local baselines | `scripts/run_baseline_experiments.py` | Developer CLI and reference flow | Not called directly from HTTP |
| Load model artifact | `mlflow.sklearn.load_model(model_uri)` (shown in `scripts/verify_mlflow_model.py`) | Returns fitted Pipeline: `predict`, `predict_proba` | Trusted model lookup, caching/version choice, error handling |
| API endpoints | `api/routes.py`, `api/main.py` | Existing skeleton | Replace mock wiring with real service |
| API JSON shapes | `api/schemas.py` | Existing schemas | Validate six inputs; define outputs including model URI/version |
| Existing API mock and tests | `api/mock_service.py`, `tests/unit/test_api.py` | Starting point | Replace mock assertions/add real API integration and HTTP tests |
| Model tracking automated tests | `tests/unit/test_mlflow_tracking.py`, `tests/integration/test_mlflow_integration.py` | Patterns for isolated SQLite tests | Add endpoint-specific, model-cache and prediction-log tests |

> **Critical distinction:** `data/loader.py` validates the **full raw training CSV**; a `/predict` request contains only the **six normalized inference fields** and should be validated by the API's own Pydantic schema. Do not require `Target`, `Failure Type`, `UDI` or `Product ID` from prediction clients.

## 8.2 Recommended `POST /train` flow

1. Validate user configuration (`model_type` as `gradient_boosting` or `xgboost`; authorized dataset/source; permitted tracking configuration). Do **not** accept arbitrary local filesystem paths or MLflow destinations from untrusted clients.
2. Load and validate the fixed initial dataset, normalize columns and call `split_features_target()`.
3. Produce `DataSplits` using `create_initial_data_splits()` (only for the initial 5,000-row case). Do not re-split saved validation/test sets when implementing retraining.
4. Call `run_training_experiment(splits, model_type, tracking_uri=server_config_uri)`.
5. Return JSON with `run_id`, `model_uri`, `model_type`, `metrics`, `status`; optional persistent job identifier if asynchronous. The tracking function already logs parameters, six metrics and the **complete model+preprocessing artifact**; the API should not create a competing second training log.
6. Return clear errors for malformed input, absent dataset, MLflow failures, invalid model type or background job failure. If training exceeds HTTP timeouts, introduce a job mechanism and `202 Accepted` + status endpoint rather than pretending immediate completion.

**Conceptual request/response** (design suggestion, not an already implemented endpoint):

```json
POST /train
{"model_type": "xgboost"}

200 OK
{
  "status": "completed",
  "model_type": "xgboost",
  "run_id": "<actual-mlflow-run-id>",
  "model_uri": "models:/m-<actual-logged-model-id>",
  "metrics": {"f1": 0.7234, "recall": 0.7083}
}
```

This example uses observed baseline metric values **only for illustration**; values from each actual run must be returned rather than hard-coded.

## 8.3 Recommended `POST /predict` flow

1. Validate one or more inference records against a Pydantic schema with **exact normalized input columns** and sensible type/range constraints; do not accept labels or leak-prone columns.
2. Select a **trusted, configured** `model_uri` (or later Registry model alias/version). An MLflow logged-model ID is not the same thing as a stable Registry version.
3. Load the complete Pipeline with `mlflow.sklearn.load_model(model_uri)` or reuse a safe cached instance. **Do not manually run feature engineering/imputation before calling the Pipeline**; it already includes all fitted transformations.
4. Build a pandas DataFrame with the six expected columns in the documented order, normalize numeric dtypes in line with the logged model signature, and compute `predict_proba(X)[:,1]`. Apply the **configured** decision threshold (currently `0.5`) for the failure flag. Do not silently assume model `.predict()` will reflect a different custom threshold.
5. Return JSON with `failure_prediction`, `failure_probability`, `model_uri` and a meaningful `model_version`/identifier (if available), plus a request/batch identifier. Validate numeric finiteness and probability bounds.
6. **Implement new persistent prediction logging** (not present in 3.1): request ID, timestamps, trusted model ID/run ID, model/version, minimal validated input fields or permitted redacted representation, probabilities, outputs, status/errors, relevant performance metadata. Design retention/access restrictions; never log credentials.
7. **Log retrievable prediction artifacts to MLflow**. For example, buffer predictions into a batch and create a dedicated MLflow prediction run with tags such as `operation=predict`, `model_uri`, `batch_id`; then use `mlflow.log_dict(payload, "predictions/predictions.json")` or `mlflow.log_artifact(csv_path, artifact_path="predictions")`. Verify the artifacts can be downloaded/retrieved from the MLflow UI. **This is Máté's new feature, not an existing capability of `run_training_experiment`.** Avoid one expensive MLflow run per single request unless volume and requirements justify it; batch/persistent queue with defined flush/retry policy is preferable.

**Example inference payload** (illustrative valid shapes; values are not an actual prediction):

```json
POST /predict
{
  "type": "M",
  "air_temperature_k": 298.4,
  "process_temperature_k": 308.3,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 120
}

200 OK
{
  "failure_prediction": 0,
  "failure_probability": 0.12,
  "model_uri": "models:/m-<configured-id>",
  "model_version": "<configured-model-version-or-identifier>",
  "request_id": "<generated-id>"
}
```

The example `0.12` is **illustrative**, not a measured score for the shown input. If model signature enforcement rejects integer-to-null conversions, fix the declared schema and add explicit tests; do not bypass validation blindly.

## 8.4 Acceptance tests Máté should add

| Area | Testable expectation |
|---|---|
| Train endpoint | Swagger/Postman call starts real fit and returns correct `run_id`, `model_uri`, status; MLflow run has parameters, six metrics and fitted Pipeline artifact |
| Predict endpoint | Valid six-feature JSON returns actual finite probability and thresholded binary prediction, with model identifier |
| API validation | Missing/unknown fields, incorrect type/ranges, NaN/non-finite values and invalid model choice produce appropriate errors |
| Model loading | Trustworthy configured model loads once/cache works; missing/malformed URI or inaccessible MLflow yields controlled errors |
| Prediction audit | Every accepted request/batch has persistent metadata with request/batch ID; failures/retries handled explicitly |
| Prediction artifacts | Aggregated JSON/CSV logged under identifiable MLflow prediction run/batch, retrievable through the UI and client |
| Test isolation | API tests use temporary SQLite MLflow, synthetic fixtures/mocks as appropriate; do not depend on developer's localhost server |
| Quality gates | `src/` coverage ≥95%; mandatory tests, Flake8 and Pyright pass in GitHub Actions |

### 8.5 Recommended integration sequence

**After 3.1 merges into `dev`:** Máté updates his API branch from `dev`; calls existing training/tracking functions; completes the API Pydantic schemas and live endpoint wiring; implements prediction request/event persistence **and** MLflow prediction artifact logging; writes integration/API tests; verifies Swagger/Postman and MLflow UI; opens a reviewed PR to `dev`. Coordinate tracking URI/environment variables and model artifact availability across machines or Docker services. Do not share local `localhost:5000` URIs as if they were remotely accessible.

---

## 9. Completion checklist / current limitations

- [x] Two baseline training models and common fitted preprocessing Pipeline.
- [x] Six validation metrics with consistent evaluation logic.
- [x] Local MLflow experiment tracking and successful model logging for both models.
- [x] Local model loading/reproducibility check demonstrated for Gradient Boosting; XGBoost described as checked.
- [x] Mocked unit coverage and SQLite-backed two-model integration tests; user-reported source coverage 99.37%.
- [ ] Verify clean requirements installation with an explicitly pinned `xgboost` dependency.
- [ ] Verify final Flake8/Pyright and all mandatory tests on the actual commit.
- [ ] Obtain green GitHub Actions PR checks, code review and merge to `dev`.
- [ ] Implement Task 4.1/10 live FastAPI wiring and prediction audit/artifact logging (Máté).
- [ ] Implement Registry selection and shared MLflow deployment in their later planned tasks.

**Ownership:** The 3.1 branch supplies reusable ML training, preprocessing, validation, model persistence and MLflow logging. The 4.1/10 API branch supplies HTTP contracts, operational model selection, validation/error handling, persistent prediction logging and verifiable prediction artifacts in MLflow.
