# MLSecOps – Preprocessing Branch: Implementation and Quality Assurance

**Branch:** `feature/preprocessing-pipeline`  
**Project:** Predictive Maintenance / MLSecOps  
**Documented status:** November 1, 2026  
**Scope:** Data loading, validation, preprocessing, splitting, class imbalance handling, persistence, and testing implemented in this branch. Model training, FastAPI, MLflow, monitoring, Docker, and DVC remote storage integration are **not yet part** of this completed branch.

## 1. Objectives and Implementation Principles

The purpose of the data processing layer is to transform the raw AI4I 2020 predictive maintenance dataset into consistent, controlled, and reproducible model inputs. Data loading, validation, feature selection, splitting, training-specific preprocessing, and class imbalance handling are implemented as separate, testable components. These components can later be reused by MLflow-based model training and FastAPI inference.

The main principles of this branch are:

- **Raw data is never overwritten**; processing operations work on copies.
- Identifier columns and columns that directly reveal the target outcome must not be included as model inputs.
- Learned preprocessing statistics are fitted **exclusively on the initial training data**.
- Validation, future, and final test partitions are handled separately; the final test set is not used for training or model selection.
- The imbalance caused by rare failure cases is intended to be addressed through weights assigned to training samples; the validation/test distributions are not changed.
- Production code is supported by unit and integration tests, linting, static type checking, and coverage measurement.

## 2. Dataset and Target Variable

**Input:** `data/raw/predictive_maintenance_5000.csv` (5,000 observations and 10 original columns). The initial dataset contains 4,837 records with `Target = 0` and 163 with `Target = 1`, so the positive class represents 3.26% of the data. The objective is binary machine failure prediction (`target`). **Model performance is outside the scope of testing in this branch**, because model training has not yet been implemented.

**Required raw columns:** `UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Target`, `Failure Type`.

**Six original features** are selected for the model: `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, and `tool_wear_min`. `UDI` and `Product ID` are record/product identifiers, while `Failure Type` describes the failure outcome and therefore poses a data leakage risk. `target` is used only as the label (`y`), not as a feature. The subsequently engineered `temperature_difference_k` increases the number of numerical features to six (five original numerical features plus one engineered feature). With one-hot encoding of `type`, the current dataset produces **nine transformed features** in total.

## 3. Created and Modified Python Modules

### 3.1. `src/predictive_maintenance/data/loader.py`

**Purpose:** `load_raw_data(data_path)` reads a CSV from the supplied path, checks that the file exists and that the CSV header contains no duplicate column names, then validates the loaded table using `validate_raw_data(df)`.

**Main checks:** A non-empty DataFrame; presence of all 10 required columns; unique column names; appropriate dtypes for numerical columns; validation of fields requiring integer values; correct dtypes for text columns; `Target ∈ {0,1}` without missing target values; and `Type ∈ {L,M,H}`. Validation uses pandas dtype checks rather than silent automatic conversion. Informative `FileNotFoundError` and `ValueError` messages help trace problems.

**Rationale:** An invalid schema should be rejected early, rather than causing a later and less understandable scikit-learn failure. The header is inspected before `read_csv` because pandas may automatically rename duplicate CSV column names. Numerical validation is handled by a separate `_validate_numeric_columns()` helper, reducing the complexity of `validate_raw_data` without changing its behavior.

### 3.2. `src/predictive_maintenance/data/preprocessing.py`

**Responsibilities:**

- `rename_columns(df)`: Applies consistent `snake_case` column names according to `COLUMN_MAPPING`; rejects empty input and duplicate column names after renaming.
- `select_features(df)`: Returns the six selected original features in a fixed order; excludes identifiers, `failure_type`, and `target`.
- `split_features_target(df)`: Produces a separate feature DataFrame (`X`) and label Series (`y`), validating the existence and binary values of the target.
- `engineer_temperature_difference(df)` / `add_engineered_features(df)`: Adds `process_temperature_k - air_temperature_k` while leaving the original DataFrame unchanged.
- `create_imputer()`: Returns a separate, unfitted imputation `ColumnTransformer` (median for numerical features and most frequent value for categorical features). This is a supporting component, **not** the complete model-input preprocessing workflow.
- `create_preprocessing_pipeline()`: Creates the complete scikit-learn `Pipeline` for feature engineering, imputation, scaling, and categorical encoding.

**Why median imputation for numerical features?** The median is less sensitive to extreme measured values than the mean. For categorical features, `most_frequent` provides a simple replacement rule learned from the training data. `StandardScaler` puts numerical features on comparable scales, supporting a wider range of future models; this choice **does not** imply that scaling is necessarily required for Gradient Boosting. `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` converts `type` categories into numerical columns suitable for machine learning and prevents an unseen category from immediately causing an inference error. The output is a pandas DataFrame to retain understandable feature names.

**Why use a Pipeline?** Learned parameters of the imputers, scaler, and encoder are kept together. Consequently, `fit` is performed only on the training partition, while `transform` is used for all other partitions. This reduces the risk of data leakage. The temperature feature engineering function is defined at module level to facilitate reproducible `joblib` serialization and deserialization.

### 3.3. `src/predictive_maintenance/data/splitting.py`

**Purpose:** `create_initial_data_splits(X, y, random_state=42)` divides the **initial dataset of exactly 5,000 records** into four partitions with disjoint indexes. The `DataSplits` dataclass stores the corresponding `X` and `y` partitions. The function validates the record count, index equality and uniqueness, and the binary target.

| Partition | Records | Share | Purpose |
|---|---:|---:|---|
| Train | 3,000 | 60% | Initial model and preprocessing fitting |
| Validation | 750 | 15% | Model selection and evaluation during development |
| Future | 750 | 15% | Simulated future observations, drift simulation, and use of **labeled** data for retraining |
| Final test | 500 | 10% | Held-out final evaluation |

**Why these proportions?** The initial training partition retains 60% of the samples while reserving separate validation, simulated future-data, and final test partitions. Keeping the future partition separate supports the assignment's drift and retraining demonstration. This is a **project design decision**, not an empirically optimized split or one that is universally best for every problem.

**Why stratification?** The positive (failure) class is rare (3.26%). With an ordinary random split, a smaller partition might contain disproportionately few positive examples. Using `stratify` preserves class proportions close to the full dataset in each partition. `random_state=42` makes the split reproducible. Tests verify partition sizes, disjoint indexes, alignment of features and labels, class proportions, and determinism.

**Limitation:** This function is intended **only for the initial split**. It must not be rerun when future observations are gradually incorporated, because the validation and final test partitions must remain fixed. In real deployments, labels for future observations may not be immediately available.

### 3.4. `src/predictive_maintenance/data/imbalance.py`

**Purpose:** `validate_binary_target(y)` checks that the target is non-empty and contains both binary classes; `create_class_distribution(y)` returns counts and percentages by class; and `calculate_sample_weights(y_train)` uses `compute_sample_weight(class_weight="balanced")` to calculate weights **only from training labels**, returning a `pd.Series` that preserves the original indexes.

**Why class/sample weighting?** Only 163 of the 5,000 observations are positive (3.26%), so without weighting, training could focus disproportionately on the common negative class. The balanced formula is `w_c = n_samples / (n_classes × n_c)`. Samples from the rarer positive class therefore receive higher weights. No SMOTE or physical oversampling is performed, so no synthetic records are generated and the original validation/test distributions remain unchanged. The weights **must actually be applied in a later model `.fit(..., sample_weight=...)` call**; this branch only calculates and tests them. For retraining, weights must be recalculated from the current training partition.

### 3.5. `src/predictive_maintenance/data/persistence.py`

**Purpose:** `save_preprocessing_pipeline(pipeline, artifact_path)` accepts only a fitted scikit-learn `Pipeline` and saves it with a `.joblib` extension. `load_preprocessing_pipeline(artifact_path)` loads the artifact, checks that the file exists, converts some corrupted-input failures into controlled exceptions, and validates the object's Pipeline type and fitted state.

**Why `joblib`?** Learned statistics of fitted scikit-learn transformers (imputer, scaler, and encoder) can be saved together consistently and reused during inference. The pickle-based format **must only be used with trusted artifacts**; loading arbitrary external files is unsafe. Unit and integration tests save artifacts under `tmp_path`, so this branch **does not contain a persistent preprocessing artifact**. Registering the planned combined model and preprocessing Pipeline in MLflow is a future task; the local persistence functions are preparatory building blocks.

## 4. Git and Repository-Wide QA

Development takes place on `feature/preprocessing-pipeline`, which is intended to be merged into `dev` through a pull request and teammate review. Production code is under `src/`, and automated tests are under `tests/unit/` and `tests/integration/`. The branch creates or modifies the following root-level QA configuration files:

- `.coveragerc`: Line coverage with a 95% minimum and an HTML report under `htmlcov/`; includes the current repository's specified Python directories (`src`, `tests`, `infra`, `app`). There is no `scripts/` directory, so it must not be included.
- `.flake8`: Maximum line length of 120 characters, `max-complexity = 10`, and targeted exclusions for virtual environments and generated files.
- `pyrightconfig.json`: Python 3.11, `standard` type-checking mode, checks for the project's Python code, and support for the `src` import path.
- `.github/workflows/python-qa.yml`: PR/push QA workflow for the `dev` and `main` branches. Under the previously agreed **temporary** approach, unit tests, coverage, Flake8, and Pyright are mandatory; integration tests run only when the CSV is available. The workflow emits a warning for a missing dataset and **does not treat skipped integration tests as passed**.

**Important:** Having the workflow file does not by itself prevent merging. The `Python QA` check must be configured as a required status check in GitHub branch rulesets/branch protection, together with the required PR review. Successful execution of the workflow on GitHub and finalization of DVC data access are still pending.

## 5. Automated Tests: Files Created and Their Purposes

### 5.1. Unit Tests

| File | Test Cases | Behavior Verified |
|---|---:|---|
| `tests/unit/test_loader.py` | 15 | Valid CSV, missing/empty files, required and duplicate columns, numeric/integer/string dtypes, invalid Target and Type |
| `tests/unit/test_splitting.py` | 13 | 3,000/750/750/500 partition sizes, retention of all records, no overlap, reproducibility, stratification, X/y index alignment, invalid inputs |
| `tests/unit/test_preprocessing.py` | 21 | Renaming, feature selection, target validation, missing/empty input, engineered temperature feature, imputation, unseen categories, nine features, scaling, train-only statistics |
| `tests/unit/test_imbalance.py` | 11 | Class distribution, weight formula, higher weight for the positive class, index alignment, total weight, changing weights during retraining, invalid target |
| `tests/unit/test_persistence.py` | 8 | Saving/loading a fitted pipeline, identical output, invalid extension, unfitted pipeline, missing/wrong-type/corrupted artifacts |
| **Unit total** | **68** | Isolated checks using small, synthetic datasets |

During the Step 12 coverage improvements, tests originally placed in a separate `test_coverage_gaps.py` file were **integrated into the appropriate existing test files**. Therefore, the final test structure contains no separate `gaps.py`.

### 5.2. Integration Tests

| File | Test Cases | Integration Verified |
|---|---:|---|
| `tests/integration/test_data_pipeline.py` | 3 | Real CSV → loader → DataProcessor → split → preprocessing fitted on train; sample weights aligned with X_train indexes; fitted pipeline save → load → identical inference output |

Integration tests require `data/raw/predictive_maintenance_5000.csv` to be available locally. Unit tests can run independently of this file.

## 6. Test and QA Results

### 6.1. Actual Results of the Full Test Suite

Most recent execution of the assembled source and test files:

```text
python -m pytest tests/ -q
71 passed
```

**71/71 tests passed**, consisting of 68 unit tests and 3 integration tests. This documented result applies to the tested preprocessing branch, **not** to the full MLflow/API/monitoring system, which has not yet been implemented.

### 6.2. Line Coverage – Actual Measured Results

Results measured using the corrected source modules and consolidated tests:

| Module under `src/predictive_maintenance/data/` | Stmts | Miss | Coverage |
|---|---:|---:|---:|
| `imbalance.py` | 18 | 0 | 100% |
| `loader.py` | 45 | 1 | 98% |
| `persistence.py` | 25 | 0 | 100% |
| `preprocessing.py` | 58 | 0 | 100% |
| `splitting.py` | 35 | 0 | 100% |
| **SRC total** | **181** | **1** | **99%** |

For the combined `src/` + `tests/` measurement, **only 1 of 641 measurable statements remained uncovered, giving an overall coverage of 99%**. Coverage measures *executed statements* (line coverage); it does not guarantee that every logical branch or every future input will work correctly. The result applies to the measured codebase and must be reassessed when new modules are added to the repository.

Commands for verification (from the project root):

```powershell
$env:PYTHONPATH = "src"
python -m coverage erase
python -m coverage run -m pytest tests/
python -m coverage report -m --fail-under=95
python -m coverage report -m --include="src/*" --fail-under=95
python -m coverage html
```

### 6.3. Flake8 and Pyright – Results

Most recent checks run in the developer's PowerShell environment:

- **Flake8:** After nine earlier warnings/errors, one `E302` remained concerning the number of blank lines before a function in `preprocessing.py`. This was corrected to two blank lines, and the developer confirmed that the change was made. **No new Flake8 output after the correction has been attached to this documentation**, so a fresh execution is required for independent confirmation.
- **Pyright:** `0 errors, 0 warnings, 0 informations` – confirmed in the developer environment.

Commands for a complete updated check:

```powershell
python -m flake8 src tests infra app
python -m pyright
```

The `scripts/` directory does not exist and has therefore been excluded from both the configuration and the execution command.

### 6.4. Important Limitations and Dependencies

1. Weighting of the positive class is currently **calculated**, not yet applied to actual model training.
2. The `future` partition contains held-out records from the 2020 dataset; it is not real, chronological production streaming data.
3. Pipeline serialization with `joblib` takes place **only in tests, in temporary directories**; MLflow artifact tracking and the Model Registry will be implemented later.
4. The GitHub Actions integration tests are currently **conditional** on the availability of the CSV. They must become mandatory after remote DVC storage and `dvc pull` have been integrated into CI.
5. The tests do not replace future integration and smoke tests of the complete Docker Compose, MLflow, FastAPI, and monitoring systems.

## 7. Dependencies and Execution Guide

The root-level `requirements.txt` lists the **direct dependencies of the currently implemented preprocessing and QA components**: NumPy, pandas, scikit-learn, joblib, pytest, Coverage.py, Flake8, and Pyright. The current file uses compatible version ranges; **it is not a fully pinned, byte-for-byte reproducible lockfile**. The exact versions in the team's tested Windows/Python 3.11 `.venv` must be recorded later; MLflow, FastAPI, Evidently, Streamlit, and DVC dependencies should be added only when those components are implemented.

```powershell
# Windows PowerShell, from the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"

# Unit tests do not require the real CSV
python -m pytest tests/unit/ -q

# The full test suite requires the raw CSV
python -m pytest tests/ -q

# QA
python -m coverage erase
python -m coverage run -m pytest tests/
python -m coverage report -m --fail-under=95
python -m coverage report -m --include="src/*" --fail-under=95
python -m flake8 src tests infra app
python -m pyright
```

**Environment-dependent requirement:** The CSV must be available at the specified relative path before running integration tests. In the future CI environment, it should be restored using DVC. If the CSV is missing, the current temporary workflow skips the integration tests and reports the omission separately.

## 8. Conclusion and Next Steps

The preprocessing branch deliberately separates and makes testable raw data validation, leakage-free feature selection, reproducible four-way stratified splitting, scikit-learn preprocessing fitted only on training data, class imbalance sample weighting, and temporary fitted-pipeline persistence/inference. **The measured 71/71 passing tests and 99% source coverage apply to the current data processing components.**

Before finalizing the pull request, verify installation from `requirements.txt` in a fresh environment, repeat all QA commands, and inspect the GitHub Actions results. Merging into `dev` should follow the team's branching rules, including teammate review and a merge commit. Future development will add model training and MLflow tracking/registry, a REST API, DVC storage, monitoring, and Docker Compose integration.
