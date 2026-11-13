# DVC–MLflow training integration

## Scope

The integration connects the DVC-managed raw dataset to preprocessing,
MLflow training, model registration and champion selection.

It records verified Git/DVC references and exact initial split membership.
API endpoint integration and Docker Compose configuration are separate tasks.

## Requirements

- Install the project dependencies and DVC with S3 support.
- Configure personal DagsHub credentials using the README instructions.
- Run commands from the repository root.
- Commit or stash all tracked and untracked changes before training.
- Provide an explicit, accessible MLflow tracking and Registry backend.

Credentials and generated artifacts must remain outside Git.

## Restore the dataset

```powershell
.\.venv\Scripts\python.exe -m dvc pull data/raw.dvc -r dagshub
```

Expected dataset:

```text
data/raw/predictive_maintenance_5000.csv
```

The controlling metadata is `data/raw.dvc`.
Its `.dir` hash identifies the entire `data/raw` directory.

## Run local training

Create an ignored directory for the local SQLite backend:

```powershell
New-Item -ItemType Directory -Force .\artifacts\dvc-lineage-check | Out-Null
```

Check that Git reports no pending changes:

```powershell
git status --short
```

Then run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts/run_dvc_training.py --model-type gradient_boosting --tracking-uri "sqlite:///artifacts/dvc-lineage-check/mlflow.db"
```

Supported model types are `gradient_boosting` and `xgboost`.

The command performs real training and creates a registered model version.
Champion promotion is conditional on the Registry selection rules.
Repeated execution creates additional runs and registered versions.

## Verification before training

The workflow requires:

- A valid Git HEAD and clean working tree.
- Valid MD5 directory metadata in `data/raw.dvc`.
- An available canonical CSV.
- An empty DVC status result for `data/raw.dvc`.

Git and DVC references are checked again after loading and splitting.
Missing, modified or unverifiable data prevents training.

These checks do not lock files against external changes during execution.

## Initial split

The current splitter requires exactly 5,000 rows and uses seed 42:

| Partition | Rows |
|---|---:|
| Train | 3,000 |
| Validation | 750 |
| Future | 750 |
| Final test | 500 |

The split policy identifier describes the configured method, seed and sizes.
The manifest records exact row indexes and their order, plus the sklearn version.

Row indexes refer to the loaded source dataset. They must be interpreted
together with its DVC identifier, not as independent permanent record IDs.

## MLflow records

The training run contains:

- `dataset_name`, `dataset_dvc_hash`, `dataset_git_commit`.
- `dataset_split_version`.
- `dataset_split_manifest_sha256`.
- `lineage/dataset_metadata.json`.
- `lineage/split_manifest.json`.

The registered model version links back through `training_run_id`.

The manifest hash is SHA-256 over UTF-8 JSON serialized with sorted keys,
compact separators and non-finite values disallowed.

Dataset metadata and the split manifest are logged before champion selection.
A logging failure prevents selection and promotion; a candidate version
may already have been registered.

Raw dataset values are not copied into the lineage artifacts.

## API handoff

The reusable entrypoint is:

```python
train_from_dvc(repository_root, model_type, tracking_uri)
```

The API should supply server-controlled repository and backend settings.
It must handle missing data, dirty source state and backend errors.
The current integration does not change HTTP routes or serving behavior.

## Automated validation

The existing Python QA workflow installs DVC, restores the shared dataset
and runs all integration tests.

DVC-specific integration tests use temporary Git repositories, a local
DVC remote and isolated SQLite MLflow backends. They verify:

- Data restoration into a fresh clone.
- Actual Git/DVC references in MLflow.
- Exact split membership and its SHA-256 identifier.
- Rejection of modified data before an MLflow training run is created.

Local validation on Windows / Python 3.11:
373 tests passed; source statement coverage 100%;
combined source and test statement coverage 99.97%.

GitHub Actions results must be checked separately on the PR.
