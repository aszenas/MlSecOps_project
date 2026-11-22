# Dataset versions and submission export

This guide describes the DVC-managed datasets in this checkout, the
reproducible preprocessed submission export, and the verified restoration of
that export. It documents the actual interfaces and identifiers in the
feature/data-release-qa revision; it does not turn the submission export into
a retraining dataset.

## Dataset roles

| Artifact | DVC metadata | Role |
| --- | --- | --- |
| Original raw dataset | data/raw.dvc | Canonical 5,000-row source used by the loader and the initial split. |
| Preprocessed submission export | data/processed/submission.dvc | Versioned, transformed export containing all four initial partitions. |
| Training v1 | data/processed/training/train_v1.csv.dvc | Separate training artifact selected by the DVC training path when dataset_version="v1". |
| Training v2 | data/processed/training/train_v2.csv.dvc | Separate, verified training artifact for the v2 training path; its rows are checked against the expected retraining split. |

The training pointers exist in this revision, but the clean-clone restoration
described below tested only data/processed/submission.dvc. It is therefore
not evidence that both training CSVs were restored in that run. The submission
export is not automatically a retraining input.

Do not commit the raw or processed data files to Git. Commit the small DVC
pointer files and keep credentials in the ignored local DVC configuration.

## Submission schema and preprocessing contract

The export contains 5,000 rows in this stable column order:

1. air_temperature_k
2. process_temperature_k
3. rotational_speed_rpm
4. torque_nm
5. tool_wear_min
6. temperature_difference_k
7. type_H
8. type_L
9. type_M
10. target
11. partition
12. source_udi

The first nine columns are the transformed model features. target is the
binary machine-failure label. partition identifies train, validation, future,
or test. source_udi is the original raw UDI value, not the loaded DataFrame
index, so each exported row can be traced back to the raw source.

The deterministic initial split uses seed 42 and has these sizes:

| Partition | Rows |
| --- | ---: |
| train | 3,000 |
| validation | 750 |
| future | 750 |
| test | 500 |

The preprocessing transformer is fitted only on train. The same fitted
transformer is used with transform for validation, future, and test. The
complete export must not be treated as one training set: it contains reserved
validation, future, and final-test rows.

## Recreate the submission export

The repository workflow uses Python 3.11 in CI. The checked export was
generated and verified with Python 3.12.10. Use the project virtual
environment and install the repository dependencies plus the pinned DVC S3
extra:

    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m pip install "dvc[s3]==3.67.1"

Configure personal DagsHub authentication in the Git-ignored
.dvc/config.local; never put a token in a command, source file, or commit.
From the repository root, restore exactly the raw directory referenced by the
current Git revision:

    .\.venv\Scripts\python.exe -m dvc pull data/raw.dvc -r dagshub
    .\.venv\Scripts\python.exe -m dvc status data/raw.dvc

The export CLI has only these path options:

    $env:PYTHONPATH = (Resolve-Path .\src).Path
    .\.venv\Scripts\python.exe scripts/export_preprocessed_submission.py --repository-root (Get-Location).Path --output-dir (Join-Path (Get-Location).Path "data/processed/submission")

The repository-root option defaults to the repository containing the script
and output-dir defaults to data/processed/submission; both are shown
explicitly above for a reproducible run. The CLI verifies the raw DVC output,
loads data/raw/predictive_maintenance_5000.csv, creates the fixed split, and
writes:

    data/processed/submission/submission_preprocessed.csv
    data/processed/submission/submission_metadata.json

The metadata records the raw source and DVC identifier, split policy,
split-manifest SHA-256, row and partition counts, feature/output columns, the
train-only fit contract, and the Python/preprocessing package versions. The
export writer is deterministic for the same raw DVC content, split policy,
code, and dependency versions.

## Restore an existing submission export

Select the Git revision whose DVC pointer is required, then restore only the
submission output from the DagsHub remote:

    git checkout 52f7c549b26ba27e1d8c56a726d830d996009d54
    .\.venv\Scripts\python.exe -m dvc pull data/processed/submission.dvc -r dagshub
    .\.venv\Scripts\python.exe -m dvc status data/processed/submission.dvc

This is a download of the versioned DVC output, not regeneration from the raw
dataset. Validate the restored files without copying them into Git:

    $csv = "data/processed/submission/submission_preprocessed.csv"
    $metadata = "data/processed/submission/submission_metadata.json"
    Get-FileHash $csv -Algorithm SHA256
    Get-FileHash $metadata -Algorithm SHA256
    python -c "import pandas as pd; df = pd.read_csv(r'$csv'); print(len(df)); print(df['partition'].value_counts().sort_index().to_dict())"
    git status --short

The expected partition result is {'future': 750, 'test': 500, 'train': 3000,
'validation': 750}. A clean target-specific DVC status and a clean Git
working tree are required after restoration.

## Verified identifiers for this revision

These values were read from the checked-in pointers and generated metadata,
not inferred from filenames:

| Identifier | Value |
| --- | --- |
| Git revision | 52f7c549b26ba27e1d8c56a726d830d996009d54 |
| Raw DVC directory hash | b740c7a32109e28365b89752a09e6096.dir |
| Submission DVC directory hash | 737d57f421900bc799f3a3997e3a4f06.dir |
| Submission CSV SHA-256 | 1e8069e500c1628505aa0a35d712627a39c5b7f9a40e368cfa14de05694646d5 |
| Submission metadata SHA-256 | 95d075eee37a40995dbf670868b4e3197a62ca46bcd3f06bfe0cab4eac8ef8a8 |
| Split policy | stratified-v1-seed42-train3000-validation750-future750-test500 |
| Split manifest SHA-256 | 1fc78ae4f79812f989588bce460e70a1074c0ab0e0e708124bdec60caa85b042 |

## Clean-clone verification

The submission restore was verified from the pushed Git revision in a separate
temporary clone. The clone started with an empty, clone-local DVC cache; the
targeted dvc pull data/processed/submission.dvc --remote dagshub populated
that cache and restored the two submission files. The restored CSV had 5,000
rows, the 3,000/750/750/500 partition sizes, and the expected 12-column schema.
Both restored file SHA-256 values matched the source export byte-for-byte. The
target-specific DVC status and the clone Git status were clean.

This verification covers the submission output only. It does not claim that
the raw dataset or either separate training artifact was restored in that
clean-clone trial.
