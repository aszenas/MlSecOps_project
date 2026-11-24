"""Load and identify the versioned reference data for monitoring."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import (
    FEATURE_COLUMNS,
    DataProcessor,
)
from predictive_maintenance.data.provenance import verify_dvc_dataset
from predictive_maintenance.data.split_manifest import build_split_manifest
from predictive_maintenance.data.splitting import create_initial_data_splits
from predictive_maintenance.models.dvc_lineage import INITIAL_SPLIT_VERSION


RAW_DATASET_RELATIVE_PATH = Path("data/raw/predictive_maintenance_5000.csv")
DVC_POINTER_RELATIVE_PATH = Path("data/raw.dvc")
NUMERIC_FEATURES = tuple(FEATURE_COLUMNS[1:])


@dataclass(frozen=True)
class MonitoringReferenceMetadata:
    """Version and split identifiers for a monitoring reference table."""

    source_file: str
    dvc_pointer: str
    raw_directory_dvc_hash: str
    split_policy: str
    reference_row_count: int
    split_manifest_sha256: str
    reference_row_hash: str


@dataclass(frozen=True)
class MonitoringReferenceResult:
    """Reference features together with identifiers needed by later monitoring steps."""

    dataframe: pd.DataFrame
    metadata: MonitoringReferenceMetadata


def _canonical_hash(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _reference_row_hash(split_manifest: dict[str, object]) -> str:
    """Hash the ordered initial-train row indexes, not row values or other partitions."""
    row_indices = split_manifest.get("row_indices")
    if not isinstance(row_indices, dict) or not isinstance(row_indices.get("train"), list):
        raise ValueError("The split manifest has no initial training row indexes.")
    return _canonical_hash({
        "schema_version": 1,
        "row_index_meaning": "index of the loaded source dataset",
        "split_policy": INITIAL_SPLIT_VERSION,
        "train_row_indices": row_indices["train"],
    })


def _reference_features(repository_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    dataset_path = repository_root / RAW_DATASET_RELATIVE_PATH
    raw_data = load_raw_data(dataset_path)

    processor = DataProcessor()
    normalized_data = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized_data)
    splits = create_initial_data_splits(features, target)
    split_manifest = build_split_manifest(splits)

    reference = splits.X_train.loc[:, list(FEATURE_COLUMNS)].copy()
    reference = reference.astype({column: "float64" for column in NUMERIC_FEATURES})
    return reference, split_manifest


def load_monitoring_reference(repository_root: Path) -> MonitoringReferenceResult:
    """Load the DVC-verified initial 3,000-row training reference.

    The raw directory hash is read and verified before loading, then verified again
    afterwards. The split-manifest hash uses the same canonical JSON serialization as
    ``log_split_manifest``; the reference-row hash uses the ordered initial-train row
    indexes from that manifest, plus its policy and row-index meaning.
    """
    repository_root = Path(repository_root).resolve()
    dvc_hash_before = verify_dvc_dataset(repository_root)
    reference, split_manifest = _reference_features(repository_root)
    dvc_hash_after = verify_dvc_dataset(repository_root)
    if dvc_hash_before != dvc_hash_after:
        raise ValueError("The DVC raw dataset changed while loading the monitoring reference.")

    split_manifest_hash = _canonical_hash(split_manifest)
    metadata = MonitoringReferenceMetadata(
        source_file=RAW_DATASET_RELATIVE_PATH.as_posix(),
        dvc_pointer=DVC_POINTER_RELATIVE_PATH.as_posix(),
        raw_directory_dvc_hash=dvc_hash_after,
        split_policy=INITIAL_SPLIT_VERSION,
        reference_row_count=len(reference),
        split_manifest_sha256=split_manifest_hash,
        reference_row_hash=_reference_row_hash(split_manifest),
    )
    return MonitoringReferenceResult(dataframe=reference, metadata=metadata)
