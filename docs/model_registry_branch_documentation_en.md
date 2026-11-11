# MLSecOps — Model Registry, Champion Selection & Dataset Lineage (3.2)

**Project:** Predictive Maintenance / MLSecOps  
**Development branch:** `feature/mlflow-model-registry` (branched from `dev`)  
**Documentation date:** 12 November 2026  
**Related documents:** `docs/mlflow_training_branch_documentation_en.md` (3.1); preprocessing branch documentation (2.1)  
**Purpose:** Document what was built and why, what is actually logged, the exact contract for the future DVC integration (covering 2.1 + 3.1 + 3.2), QA/CI changes, and the handoff to Máté's FastAPI implementation.

> **Evidence/status convention:** **Implemented** = represented by the supplied 3.2 modules and tests. **Locally verified** = developer reports the complete test suite, coverage gate, Flake8 and Pyright are green after committing. **Proposed / integration pending** = instructions and contracts for changes **not yet wired**. The live `dev` branch, DVC configuration, current QA runner, Máté's API files and GitHub Actions were not inspected here; paths referring to them require verification at merge time. A local test pass does not establish a successful PR, remote CI, production readiness or DVC end-to-end execution.

## 1. Scope, branch work and rationale

3.1 already trains Gradient Boosting and XGBoost, fits an **entire preprocessing + classifier** sklearn Pipeline on training rows, evaluates six validation metrics, and logs MLflow runs/model artifacts. However, a logged model (`models:/m-...` with MLflow 3) is not, on its own, a named registered version or deployable champion. Step 3.2 adds the controlled selection and audit layer.

| Implemented component | File | Why it was needed |
|---|---|---|
| Model registration and aliases | `src/predictive_maintenance/models/model_registry.py` | Preserve immutable model versions; set/read the `champion` alias; load a trusted, fitted Pipeline independently of training. |
| Champion comparison | `src/predictive_maintenance/models/model_selection.py` | Compare challenger vs current champion on **the same held-out validation set** and threshold; enforce an explicit safety gate. |
| Decision/promotion separation | `model_selection.py`, `training_registry.py` | `apply_promotion=False` computes eligibility before changing the Registry; avoids promoting before audit. |
| Training–Registry orchestrator | `src/predictive_maintenance/models/training_registry.py` | One callable joins the existing 3.1 training, registration, comparison, audit and verified alias switch. |
| Persisted audit | `training_registry.py` | Record *why* promotion was approved/rejected, model/run linkage, challenger/champion metrics and status transitions. |
| Optional dataset lineage | `src/predictive_maintenance/models/data_lineage.py` | Prepare reproducible references to future DVC-managed data **without waiting** for DVC integration or duplicating CSVs in MLflow. |
| Tests | `tests/unit/test_model_registry.py`, `test_model_selection.py`, `test_training_registry.py`, `test_data_lineage.py`, `test_training_registry_lineage.py`; `tests/integration/test_model_registry_integration.py`, `test_training_registry_integration.py` | Mocked edge/failure paths and real MLflow SQLite round trips including versions, tags, JSON artifact, champion reload and repeated training. |

**Out of scope / not yet implemented:** automatic `dvc pull` and DVC content identification; a DVC-driven train command; CLI/HTTP invocation of the new orchestrator from Máté's API; prediction-event logs; distributed promotion locking or atomic alias+audit transaction; live serving/deployment; a shared authenticated MLflow backend.

## 2. Relevant 2.1 / 3.1 / 3.2 architecture

```text
DVC remote (FUTURE integration)
    -> dvc pull / dvc repro (FUTURE integration)
    -> data/raw/predictive_maintenance_5000.csv
    -> data/loader.py: load_raw_data(path)             [2.1]
    -> data/preprocessing.py: DataProcessor             [2.1]
         rename_columns -> split_features_target
    -> data/splitting.py: create_initial_data_splits    [2.1]
         DataSplits: X/y train | validation | future | test
    -> models/training_registry.py: train_register_and_select [3.2]
         -> models/mlflow_tracking.py: run_training_experiment [3.1]
              -> models/training.py: fitted preprocessing+model Pipeline
              -> models/evaluation.py: six validation metrics
              -> MLflow training run + logged model artifact
         -> models/model_registry.py: register_model -> version N
         -> models/data_lineage.py: optional JSON + tags
         -> models/model_selection.py: compare against champion
         -> training_registry.py: pre-promotion audit
         -> model_registry.py: set champion alias
         -> training_registry.py: verify alias and record status
    -> Máté's future /train and /predict API integration
```

The **existing initial split** requires exactly 5,000 rows with matching unique indices and both target classes; `random_state=42`; **3,000 train / 750 validation / 750 future / 500 final test**. Do not silently apply this function to an arbitrarily sized DVC dataset. Treat `future` as reserved for simulation/drift and `test` as final holdout; neither is used to select champion. The six raw inference columns are `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`. Exclude `udi`, `product_id`, `failure_type` and `target` from model features. `failure_type` in particular leaks the label.

## 3. Public Python contracts (implemented)

```python
from predictive_maintenance.models.model_registry import (
    REGISTERED_MODEL_NAME,  # "predictive-maintenance-model"
    CHAMPION_ALIAS,         # "champion"
    register_model,
    set_champion_version,
    get_champion_version,
    load_champion_model,
)
from predictive_maintenance.models.model_selection import (
    MINIMUM_RECALL,         # 0.70
    evaluate_and_select_champion,
)
from predictive_maintenance.models.data_lineage import DatasetLineage
from predictive_maintenance.models.training_registry import (
    train_register_and_select,
    TrainingRegistryResult,
)
```

Main orchestration contract:

```python
train_register_and_select(
    splits: DataSplits,
    model_type: ModelType,               # "gradient_boosting" | "xgboost"
    tracking_uri: str | None = None,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    minimum_recall: float = 0.70,
    threshold: float = 0.5,
    dataset_lineage: DatasetLineage | None = None,
) -> TrainingRegistryResult
```

`TrainingRegistryResult` fields: `run_id`, `model_uri`, `registered_model_name`, `registered_version`, `selection`. `selection` is a `SelectionResult` with `promoted`, `reason`, `selected_version`, `challenger_version`, `champion_version_before`, `challenger_metrics`, `champion_metrics`. **Important:** `SelectionResult.promoted` represents an approved recommendation when `evaluate_and_select_champion(..., apply_promotion=False)` is used; only verified `promotion_status=completed` plus the actual alias demonstrates a completed switch. The orchestration function is **synchronous** and alters process-level MLflow tracking/registry URI settings. Coordinate concurrency/configuration at the caller.

### 3.1 Selection rule and rationale

1. Both models are evaluated on the **same current validation DataFrame/labels** and probability threshold (default `0.5`).
2. Reject the challenger when `recall < 0.70` (exactly `0.70` qualifies).
3. If no champion exists and challenger passes recall, it is eligible to become the first champion.
4. If a champion exists, promote only when challenger `F1 > champion F1` **strictly**; equal or worse F1 leaves the alias unchanged.
5. Register challengers regardless of the outcome, preserving rejected versions for comparison and rollback.

F1 balances precision/recall; the **70% recall floor is an explicit project/business assumption**, *not* an external safety standard or proven acceptable miss rate. On a representative labelled set, 70% recall means about 30% of actual failures may be missed; review with domain stakeholders before deployment. AP and ROC-AUC remain useful diagnostic metrics but are **not** promotion gates in this implementation. When introducing new datasets, do not automatically compare F1 scores across different validation cohorts; preserve a fixed, versioned evaluation reference or document cohort changes explicitly.

### 3.2 Audit-first promotion, limitations

```text
train/log model -> register challenger -> optional lineage artifact/tags
  -> evaluate (DRY RUN) -> log selection tags + four comparison metrics
  -> if rejected: not_promoted; champion unchanged
  -> if approved: pending -> recheck current alias -> set alias
       -> read alias back -> completed
       -> on errors: attempt verification_required, then re-raise
```

`training_registry.py` factors this into `_log_selection_audit()`, `_record_promotion_status()` and `_apply_audited_promotion()`, avoiding a Flake8 C901 complex orchestration method. Initial parameter validation rejects blank model name/tracking URI and invalid recall/threshold bounds. An error in optional lineage writing aborts selection/promotion; an audit write error before alias modification prevents promotion.

**Not atomic:** alias update and MLflow tag writes are separate operations; final status logging can fail after the alias changes. `verification_required` means **inspect the actual Registry alias**; it does not necessarily mean rollback. A later audit reconciliation utility, retry/idempotency strategy and single-writer lock or transaction-capable promotion service are recommended for concurrency. The current pre-promotion alias check reduces stale-decision risk but is not a distributed compare-and-swap lock. `get_champion_version()`/`load_champion_model()` reflect the alias's current state only. A request serving from a cached model may use an older version until refreshed.

## 4. Exactly what MLflow stores TODAY

| Scope | Concrete fields / contents | Source | Stored where |
|---|---|---|---|
| Training | Model name/type, training hyperparameters, run tags and model artifact | `models/mlflow_tracking.py`, `training.py` (3.1) | Training run params/tags; logged Pipeline model artifact |
| Quality | Validation `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `average_precision` | `models/evaluation.py`, `mlflow_tracking.py` (3.1) | Training run metrics |
| Registry identity | Registered model `predictive-maintenance-model`, version, alias `champion` | `model_registry.py` | Model Registry |
| Provenance link | `training_run_id`, `model_type` | `training_registry.py` | Model version tags |
| Selection rationale | `selection_status` (`approved`/`rejected`), `selection_reason`, registered name/version, `champion_version_before`, `selection_minimum_recall`, `selection_threshold` | `_log_selection_audit()` | Training run tags; only status+reason also on model version |
| Direct comparison | `selection_challenger_f1`, `selection_challenger_recall`, and **if prior champion** `selection_champion_f1`, `selection_champion_recall` | `_log_selection_audit()` | Training run metrics |
| Actual promotion state | `promotion_status`: `pending`, `completed`, `not_promoted`, `verification_required` | `_record_promotion_status()` | Training run + challenger model-version tags |
| Dataset lineage **if supplied** | `dataset_name`; optional `dataset_dvc_hash`, `dataset_git_commit`, `dataset_split_version`; complete `dataset_metadata.json` with `split_rows` | `models/data_lineage.py` | Training run tags and `lineage/dataset_metadata.json` artifact |

Lineage is **optional**: `dataset_lineage=None` means **NO lineage artifact and no `dataset_*` tags**. `DatasetLineage` is currently:

```python
DatasetLineage(
    dataset_name="predictive_maintenance_5000",
    dvc_hash=None,           # replace with verified actual DVC hash
    git_commit=None,         # replace with actual Git SHA
    split_version="initial-seed-42",  # use only if the split policy matches
)
```

`log_dataset_lineage(client, run_id, splits, lineage)` validates nonblank strings, writes the JSON artifact **first**, then searchable run tags; split counts come from the real `DataSplits` object. This implementation does **not** auto-run `git rev-parse`, query DVC, hash CSVs, or store individual row IDs. It **does not** put the dataset lineage on the registered model version: follow `training_run_id` to the run. The example hashes in synthetic integration tests are fixtures, **not evidence of a connected DVC remote**. CSV bytes are not copied into MLflow.

### 4.1 Logging that remains to be added

| Item | Reason | Owner / future location |
|---|---|---|
| Verified Git SHA and DVC content identifier collected at training entrypoint | Reconstruct exact code/data state | DVC/CLI integration; pass `DatasetLineage(...)` to `train_register_and_select()` |
| Data file/path and selected DVC remote/config identity; resolved data object `.dvc` or `dvc.lock` reference | Distinguish multiple datasets and reproduction sources | Future data manifest / lineage schema extension |
| Split IDs or index manifest, label distribution, feature schema, preprocessing config/hash and split policy/version | Same 5,000 rows + row ordering are not implied by `train_rows` | Dataset preparation + separate immutable manifest; optional additional JSON artifact |
| MLflow structured dataset inputs (`mlflow.log_input` / client dataset input) | Optional searchable lineage in MLflow dataset UI | Future 3.1 logging improvement; avoid full raw upload |
| Persisted reason/error code for early training/registration failure | Auditable unsuccessful workflow attempts | Orchestrator/job tracking, where a run is available |
| Prediction inputs metadata, model version, thresholds, probabilities, outputs, request ID, timestamps and error codes | Traceability of API inference | Máté's `/predict` service, not the training Registry module |
| Drift/retraining linkage and champion history | Trace which alert caused which run and alias transition | Monitoring + future workflow orchestration |

Store large/versioned CSV and any generated split files in **DVC remote**; small immutable manifests/configuration under Git/DVC as appropriate; metrics/tags/JSON lineage and trained model in **MLflow**. Do not send entire raw prediction payloads to shared tracking by default; minimize sensitive data and define retention/access policies.

## 5. DVC integration — prerequisite handoff from Beni

**This entire section is an integration plan, NOT deployed code.** DVC can version raw data with `.dvc` metadata, describe workflows in `dvc.yaml`, lock file contents/dependencies with `dvc.lock`, and reproduce stages with `dvc repro`. `dvc pull` restores tracked content but is **not implicitly run by `dvc repro`** unless explicitly enabled/configured.

### 5.1 What Beni / team must provide or verify first

1. **Actual repository-level DVC layout:** `.dvc/config` tracked in Git, `.dvc/config.local` credentials excluded; selected remote name/type/storage location (e.g. shared bucket or local shared storage), access instructions for each developer and CI; check `.dvcignore`/`.gitignore`.
2. **Tracked raw data:** the agreed canonical `data/raw/predictive_maintenance_5000.csv` is retrievable with `dvc pull`; identify the controlling `data/raw/predictive_maintenance_5000.csv.dvc` file **or** a parent-directory DVC output / `dvc.lock` entry. Do **not** assume both or hardcode a particular `.dvc` path before inspection.
3. **Git-tracked metadata:** `.dvc` file(s), `.dvc/config` and any `dvc.yaml`, `dvc.lock`, `params.yaml` committed; raw CSV not duplicated as Git-tracked content. Document which hash corresponds to which content and how to retrieve it. `dvc status` clean and `dvc status -c`/`dvc pull` working in a fresh clone.
4. **DVC version / dependencies:** `dvc` and required storage extras (`dvc[s3]`, `dvc[azure]`, etc. only as appropriate) in the chosen install/lock process; credentials through environment/CI secrets, never committed.
5. **Split contract:** Do we DVC-track only raw CSV and recreate the **initial** split deterministically (`seed=42`, pinned code/library versions, preserved row order), or will DVC also track frozen split files/index manifests? **Recommendation: version split IDs/indices** to ensure fair comparison and prove identical validation cohort across runs.
6. **Trigger semantics:** Decide whether DVC only provides data (`dvc pull` + existing Python entrypoint), or also orchestrates reproducible `prepare`/`train` stages (`dvc.yaml`). These are **two separate levels of integration** and need not be implemented in one PR.
7. **Common backend:** shared MLflow tracking URI, Registry URI, artifact location and permissions accessible to DVC runner and API container. Local `http://127.0.0.1:5000` refers to the process's own machine/container, not a teammate's host.

### 5.2 Required code and configuration changes — file by file

| Path / component | Existing vs required action | Exact integration responsibility |
|---|---|---|
| `.dvc/config`, `.dvc/config.local`, `.dvcignore` | **Inspect/change with Beni** | Remote and storage settings; local secrets remain untracked. |
| `data/raw/predictive_maintenance_5000.csv.dvc` **or** DVC-tracked directory / `dvc.lock` entry | **Confirm actual controlling metadata** | Pin reproducible data object hash; `dvc pull` must restore `data/raw/...csv`. |
| `.gitignore`, `.dvcignore` | **Verify** | Ignore raw binary/CSV contents in Git while tracking metadata; avoid accidentally ignoring required `.dvc` files. |
| `requirements.txt` or dependency lock/CI install | **Review/update** | Install compatible Python, `dvc` + storage extra, MLflow, XGBoost and QA tools; pin/test versions. |
| `src/predictive_maintenance/data/loader.py` | **Reuse; path contract check** | `load_raw_data(path)` should load the DVC-restored raw file; do not fork validation logic. |
| `src/predictive_maintenance/data/preprocessing.py` | **Reuse; verify assumptions** | `DataProcessor.rename_columns`, `split_features_target`, feature pipeline; no target leakage. |
| `src/predictive_maintenance/data/splitting.py` | **Modify ONLY if datasets change** | Current exact-5000 initial split must not silently become a generic retraining splitter. If new-size/time-series data arrives, implement and test separate split policy; freeze validation/test. |
| `src/predictive_maintenance/models/training.py` | **Reuse** | Model/pipeline fitting and train-only sample weights already done in 3.1. |
| `src/predictive_maintenance/models/mlflow_tracking.py` | **Reuse; optionally enhance** | Existing experiment/run logging; optional add training input hash, signature or structured dataset input. Avoid duplicate MLflow runs. |
| `src/predictive_maintenance/models/data_lineage.py` | **Extend when exact metadata is known** | Current `DatasetLineage(dataset_name,dvc_hash,git_commit,split_version)` receives verified IDs, not placeholders; consider raw DVC path/remote/split manifest hashes. |
| `src/predictive_maintenance/models/training_registry.py` | **No signature change needed now** | Pass `dataset_lineage=...` into existing `train_register_and_select(...)`; artifact is logged before selection and promotion. |
| `src/predictive_maintenance/models/model_selection.py` / `model_registry.py` | **Reuse; review fixed evaluation reference** | Champion rule remains F1 + recall floor; lock/serialize promotion if concurrent jobs may occur. |
| **New** `scripts/run_dvc_training.py` (suggested) | **CREATE** | One official runner: verify expected data exists; load -> preprocess -> split -> obtain Git/DVC metadata -> call `train_register_and_select()` -> print machine-readable run/version/selection output; exit nonzero on failures. |
| **Optional** `dvc.yaml`, `params.yaml`, `dvc.lock` | **CREATE/EXTEND, agree with Beni** | Reproducible `prepare` stage and, **only if desired**, `train` stage; declare input/code/parameter dependencies and appropriate durable outputs; inspect exact cache/re-run semantics. |
| `tests/unit/test_data_lineage.py`, `test_training_registry_lineage.py` | **Extend** | Mock valid/invalid optional metadata and abort-before-promotion behavior. |
| **New** `tests/integration/test_dvc_training_integration.py` (suggested) | **CREATE** | Fresh workspace/temporary remote or controlled fixture: `dvc pull`, verify canonical data hash, run entrypoint, assert MLflow lineage and champion behavior; skip only if declared optional. |
| `tests/integration/test_training_registry_integration.py` | **Keep always running** | Synthetic SQLite MLflow test requires **no DVC remote**, so it should not be skipped due to missing CSV. |
| QA runner (`scripts/...`, PowerShell wrapper, `qa-plan.md`; **exact filename not verified**) | **Inspect/modify after merge** | Add a separate DVC-enabled tier; propagate failing exit codes; do not conditionally exclude ordinary SQLite integration tests. |
| `.github/workflows/python-qa.yml` (verify path) | **Modify** | Install DVC extras, provide remote credentials via secrets when appropriate; run fast tests always, DVC end-to-end job with data when available; enforce Flake8/Pyright/coverage. |
| `src/predictive_maintenance/api/{main,routes,schemas,mock_service}.py` (verify actual names) | **Máté owns** | Wire `/train` to official prepared data + orchestration and `/predict` to trusted champion Pipeline. |
| `docs/` | **Update** | Shared runbook for remote setup, CLI, artifacts, test tiers, API contract and reproducibility. |

### 5.3 Suggested DVC-to-MLflow runner contract (NOT yet implemented)

A single adapter should own dataset loading and lineage resolution; existing models must not call `dvc` themselves:

```python
# PSEUDOCODE: implement only after verifying repository DVC structure.
raw_path = Path("data/raw/predictive_maintenance_5000.csv")
assert raw_path.is_file()  # after `dvc pull` in caller/CI
raw = load_raw_data(raw_path)
processor = DataProcessor()
normalized = processor.rename_columns(raw)
X, y = processor.split_features_target(normalized)
splits = create_initial_data_splits(X, y, random_state=42)

lineage = DatasetLineage(
    dataset_name="predictive_maintenance_5000",
    dvc_hash=read_verified_dvc_output_hash(...),  # new adapter
    git_commit=read_git_head_sha(...),             # new adapter
    split_version=resolve_frozen_split_id(...),   # new adapter
)
result = train_register_and_select(
    splits=splits,
    model_type="xgboost",
    tracking_uri=server_configured_mlflow_uri,
    dataset_lineage=lineage,
)
```

`read_verified_dvc_output_hash` and `resolve_frozen_split_id` are **proposed helper functions, not existing project APIs**. The exact DVC metadata representation may be `.dvc` metadata vs `dvc.lock`; parse it using a tested DVC command/API or a documented schema, **not** an assumed file path or an arbitrary fake hash. Retrieve Git SHA using `git rev-parse HEAD` **at run time**, ensuring it corresponds to the code actually used. If SHA or DVC hash is required by a strict reproducibility run, **fail before model training** when it is not available; do not pass `None` and label that run DVC-reproducible. For the current non-DVC workflow, metadata remains optional.

**Minimal `dvc.yaml` idea (illustrative only; agree on outputs, command and actual paths first):**

```yaml
stages:
  prepare:
    cmd: python scripts/prepare_dataset.py
    deps:
      - data/raw/predictive_maintenance_5000.csv
      - scripts/prepare_dataset.py
      - src/predictive_maintenance/data/preprocessing.py
      - src/predictive_maintenance/data/splitting.py
    outs:
      - data/processed/split_manifest.json
```

`prepare_dataset.py` and `split_manifest.json` are **new suggestions**. If adding a `train` stage, note that DVC is output/dependency-oriented whereas MLflow run creation and alias mutation are external side effects; DVC run caching can **skip training/promotion when dependencies have not changed**, and it does **not** provide exactly-once Registry semantics. Prefer explicitly invoking the training CLI for deliberate champion comparisons, or design a DVC stage with an actual durable output (`training_result.json`) and explicitly define re-run/idempotency behavior. Avoid declaring MLflow's external server itself as a DVC output.

### 5.4 Suggested integration sequence and acceptance criteria

1. In the merged `dev` branch, inspect Beni's actual `.dvc`, `.dvc`-tracked files, remote/access, `.gitignore`, `dvc.yaml`/`dvc.lock` and QA runner. Agree on a single raw file path and artifact storage. Confirm the source data is **not** Git-tracked twice.
2. In a **fresh clone** with authorized credentials, run `dvc pull` and verify the CSV is present and loader validation passes; no hardcoded Windows absolute paths.
3. Establish frozen initial split IDs/version and ensure the same validation cohort is used to compare incumbent/challenger; keep test set untouched. For future data, design a **new** split policy rather than calling fixed-5000 splitter.
4. Build a tested Git/DVC metadata adapter and official CLI; pass the real `DatasetLineage` object to `train_register_and_select`.
5. Verify the MLflow run contains `dataset_*` tags and `lineage/dataset_metadata.json` **and they match the actual DVC content and split manifest**; registered version links back using `training_run_id`.
6. Verify first champion, improved challenger, recall-failing challenger, equal-F1 challenger, alias restoration/verification, and failure-before-promotion on lineage/audit logging errors.
7. Update QA runners/CI tiers; rerun local and clean CI tests; document DVC/MLflow permissions, remote, runtime config and how to reproduce one identified model version.

## 6. QA and CI test strategy, including the currently missing DVC tests

**Reported by developer:** all latest tests passed, full Flake8 and Pyright were clean, and the project coverage gate passed after the lineage changes. **Exact counts and current percentage were not provided in this exchange; do not invent them.**

```powershell
$env:PYTHONPATH = "src"
python -m flake8 src tests infra app scripts
python -m pyright
python -m coverage erase
python -m coverage run -m pytest tests/ -q
python -m coverage report -m --include="src/*" --fail-under=95
```

**Proposed test tiers:**

| Tier | Example tests / commands | Dataset/remote requirement | CI expectation |
|---|---|---|---|
| Unit | `pytest tests/unit/` | None; mocks | **Always** |
| Synthetic MLflow integration | `test_mlflow_integration.py`, `test_model_registry_integration.py`, `test_training_registry_integration.py` | Temporary SQLite + synthetic data; **no DVC** | **Always** |
| Existing real-CSV integration | E.g. `test_model_training.py`, `test_data_pipeline.py` depending on actual collection | Fixed AI4I CSV | Run **after** verified `dvc pull`; until then explicit skip/xfail with reason, not silent omission |
| **Future** DVC end-to-end | `test_dvc_training_integration.py` (not yet created) | DVC remote or isolated fake remote + known fixture; credentials if shared | Dedicated opt-in local job and CI job when configured; never silently pretend it ran |
| FastAPI | Máté's endpoint tests and MLflow audit artifact tests | Synthetic SQLite with mocked/controlled data; optional dedicated live DVC job | **Always** for normal endpoint tests |

**The QA Python runner:** its real filename/content has **not been supplied**; therefore no exact edit line can be asserted. Once inspected, update its test discovery/selection, dependency bootstrap and exit-code handling; ensure any `pytest -m "not dvc"` selection has a *separate positive DVC invocation*, or the DVC test will never run. Suggested explicit marker in `pytest.ini`:

```ini
[pytest]
markers =
    dvc: tests requiring the DVC CLI / remote (separate execution tier)
```

**Important:** merge this into existing `pytest.ini` rather than replacing its other options/markers. Mark only genuine remote-dependent tests. Prefer an isolated temporary/local DVC remote fixture for fast CI coverage; shared remote credentials should be optional and protected. Do not enable `--ignore` for the entire `tests/integration` directory just because the original real CSV is unavailable. A green unit/SQLite job with the DVC job skipped means **DVC end-to-end remains unverified**.

For a deliberate DVC run (after setup), a proposed command sequence is:

```powershell
python -m dvc pull
python -m pytest tests/integration/test_dvc_training_integration.py -m dvc -v
python -m coverage erase
python -m coverage run -m pytest tests/ -m "not dvc" -q
python -m coverage report -m --include="src/*" --fail-under=95
```

For full coverage across tiers, use `coverage run --append ...` to accumulate executed tests **in the same code checkout**; otherwise `coverage erase` between tiers resets the measurement. Assert that CI jobs publish whether the DVC tier actually ran, how many tests it executed, and that no dataset secrets leaked into logs.

## 7. Exact handoff to Máté: API integration

**Ownership:** the Registry and data pipeline modules provide reusable Python functions; Máté owns HTTP interface, FastAPI schemas/routes, request/prediction audit, error handling, operational execution, and API tests. The new 3.2 orchestrator **supersedes a direct call to** `run_training_experiment` as the preferred `/train` service operation **when automatic registration and champion evaluation are required**.

### 7.1 Files/functions to wire

| API responsibility | Existing function / file to consume | What Máté must implement |
|---|---|---|
| Training data acquisition | `data/loader.py:load_raw_data` | Read the **authorized DVC-restored CSV** (not arbitrary user-provided path); clear errors for unavailable version. |
| Normalize + target extraction | `data/preprocessing.py:DataProcessor` | Call `rename_columns`, `split_features_target`; preserve six feature names. |
| Initial split | `data/splitting.py:create_initial_data_splits` | Create fixed 5000-row `DataSplits`; separate future retraining strategy for other data sizes. |
| Training + MLflow + Registry | `models/training_registry.py:train_register_and_select` | Invoke with `splits`, `model_type`, server-controlled tracking URI, approved thresholds and verified `dataset_lineage`. |
| DVC data provenance | `models/data_lineage.py:DatasetLineage` | Hand in `dataset_name`, verified DVC hash, Git SHA, frozen split version as available; do **not** invent missing hashes. |
| Training response | `TrainingRegistryResult` + nested `SelectionResult` | Return run ID, model URI, registered version, selection reason, `promotion_status` **read from MLflow** and current champion alias; do not infer completed promotion only from `selection.promoted`. |
| Loading fitted model | `models/model_registry.py:load_champion_model` | Load trusted `models:/predictive-maintenance-model@champion` Pipeline at serving startup/refresh or call `get_champion_version` to resolve explicit version; carefully manage cache refresh. |
| Prediction | Loaded Pipeline `.predict_proba(features)` / `.predict(features)` | Pydantic schema for six raw normalized fields; create correct Pandas DataFrame, preserve units/dtypes and column order; compute prediction using agreed threshold; **never refit preprocessing**. |
| Prediction logging | **Not implemented by 3.2** | Persist request ID, timestamp, model/version, threshold, probability, class and error/status. If project spec requires request/result JSON/CSV artifacts in MLflow, implement with privacy and retention safeguards; avoid exposing full raw data by default. |
| HTTP implementation | `src/predictive_maintenance/api/main.py`, `routes.py`, `schemas.py`, `mock_service.py` **names from 3.1 docs; verify branch** | Replace mock wiring, implement Swagger/Postman-visible `/train` and `/predict`, correct status codes and endpoint tests. |
| Integration tests | Existing `tests/integration/test_training_registry_integration.py` as pattern | Add FastAPI TestClient tests: untrained/no champion, valid single/batch input, registry unavailable, unsupported model type, no data, permission error, `verification_required`, stable served version. |

### 7.2 Suggested HTTP contract (DESIGN, not existing code)

```json
POST /train
{
  "model_type": "xgboost",
  "dataset_version": "approved-server-side-version"
}

200 OK (synchronous small demo; for longer jobs prefer 202 + job ID)
{
  "run_id": "<actual run ID>",
  "registered_model_name": "predictive-maintenance-model",
  "registered_version": "<actual version>",
  "selection_reason": "higher_f1",
  "promotion_status": "completed",
  "champion_version": "<actual version read from Registry>"
}
```

These are **placeholder schemas** only. Resolve the dataset version from server-side approved sources. Never expose unrestricted file paths, tracking URIs or alias names as untrusted request parameters. Synchronous training may take too long for HTTP/containers; use `202 Accepted` + background worker/status endpoint where required. On a rejected challenger, response should distinguish `selection_status=rejected` and `promotion_status=not_promoted` and return the **current champion**, not the rejected version.

```json
POST /predict
{
  "type": "M",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 120
}

200 OK
{
  "prediction": 0,
  "failure_probability": 0.12,
  "model_version": "<actually served version>",
  "threshold": 0.5
}
```

For `/predict`, the six columns above are the inference schema; raw training file columns such as `target` or `failure_type` are **not** client inputs. If dynamically changing the threshold, do not assume `Pipeline.predict()` uses the same threshold; derive class from `predict_proba` with the configured threshold. Ensure model cache and reported `model_version` are consistent. Handle missing champion with a service-level error (e.g. 503) and validation errors with 422. `cloudpickle`/pickle model loads are trusted-only operations.

### 7.3 Precise integration contract to hand over

- MLflow tracking URI and Registry URI (shared across trainer and API), credentials, model name/alias, artifact store access.
- DVC data path and access policy; exact approved DVC/Git hash resolution procedure; frozen initial split contract.
- `train_register_and_select()` function signature and return dataclass; `DatasetLineage` schema; champion selection rule (F1 strictly higher, recall >=0.70, threshold 0.5).
- Audit tag keys and `promotion_status` meaning; errors are not equivalent to rollback.
- Trusted champion loader and six-feature JSON schema; cache refresh/version pinning expectations.
- Required Swagger/Postman demonstrations, MLflow training+prediction audit evidence, endpoint unit/integration tests and separate DVC CI tier.

## 8. Final readiness / handoff checklist

- [x] 3.2 registration, aliases, champion model loading, selection rule and dry-run mode implemented.
- [x] Training/Registry orchestration and audit-first verified promotion implemented.
- [x] Optional `DatasetLineage` tags + `lineage/dataset_metadata.json` implemented.
- [x] Developer reports green local pytest, Flake8, Pyright and >=95% coverage gate after changes.
- [ ] Confirm latest changes are pushed, PR reviewed, CI checks green and merged into `dev` (not verified in this document).
- [ ] Confirm Beni's actual DVC remote/config, tracked data and split version from a fresh clone.
- [ ] Build DVC CLI/metadata resolver and run real-data DVC->preprocessing->3.1->3.2 integration test.
- [ ] Update **actual** QA Python runner and `.github/workflows/python-qa.yml` to run and report the DVC-specific tier.
- [ ] Wire Máté's live `/train` and `/predict` endpoints and inference audit artifacts.
- [ ] Agree on shared MLflow backend, permissions, concurrency/rollback policy and deployment/monitoring boundary.

---

**Branch deliverable:** 3.2 is ready for documentation and PR **within its implemented scope**, assuming the reported local QA remains green in the commit intended for merge. **DVC/CLI/API/CI deployment are explicitly separate, outstanding integration tasks.**
