"""Export a reproducibly preprocessed submission dataset."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import cast

import pandas as pd
import sklearn

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.provenance import verify_dvc_dataset
from predictive_maintenance.data.split_manifest import build_split_manifest
from predictive_maintenance.models.dvc_lineage import INITIAL_SPLIT_VERSION
from predictive_maintenance.data.splitting import (
    DataSplits,
    RANDOM_STATE,
    create_initial_data_splits,
)


RAW_DATA_RELATIVE_PATH = Path("data/raw/predictive_maintenance_5000.csv")
EXPORT_CSV_NAME = "submission_preprocessed.csv"
EXPORT_METADATA_NAME = "submission_metadata.json"
EXPORT_SCHEMA_VERSION = 1
PACKAGE_NAMES = ("numpy", "pandas", "scikit-learn", "joblib")


@dataclass(frozen=True)
class SubmissionExportResult:
    """Describe the deterministic submission export artifacts."""

    csv_path: Path
    metadata_path: Path
    row_count: int
    partition_counts: dict[str, int]
    feature_names: tuple[str, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible result without changing the result."""
        return {
            "csv_path": str(self.csv_path),
            "metadata_path": str(self.metadata_path),
            "row_count": self.row_count,
            "partition_counts": dict(self.partition_counts),
            "feature_names": list(self.feature_names),
            "metadata": dict(self.metadata),
        }


def export_preprocessed_submission(
    repository_root: Path,
    output_dir: Path,
) -> SubmissionExportResult:
    """Export all four initial partitions with a train-only preprocessing fit."""
    root = repository_root.resolve()
    output = output_dir.resolve()
    dvc_hash = verify_dvc_dataset(root)

    raw_data = load_raw_data(root / RAW_DATA_RELATIVE_PATH)
    source_udi = cast(pd.Series, raw_data["UDI"])
    if not bool(source_udi.is_unique) or bool(source_udi.isna().any()):
        raise ValueError("The raw UDI source identifiers must be unique and non-missing.")

    processor = DataProcessor()
    normalized = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized)
    splits = create_initial_data_splits(features, target, random_state=RANDOM_STATE)

    pipeline = processor.create_preprocessing_pipeline()
    pipeline.fit(splits.X_train)

    transformed: list[pd.DataFrame] = []
    partition_counts: dict[str, int] = {}
    feature_names: tuple[str, ...] | None = None
    for partition_name, partition_features, partition_target in _partitions(splits):
        partition_result = pipeline.transform(partition_features)
        if not isinstance(partition_result, pd.DataFrame):
            raise TypeError("The preprocessing pipeline must return a DataFrame.")

        current_names = tuple(str(name) for name in partition_result.columns)
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            raise ValueError("Preprocessing produced inconsistent feature columns.")

        exported = partition_result.copy()
        exported["target"] = partition_target.to_numpy()
        exported["partition"] = partition_name
        exported["source_udi"] = source_udi.loc[partition_features.index].to_numpy()
        transformed.append(exported)
        partition_counts[partition_name] = len(exported)

    if feature_names is None:
        raise ValueError("No partitions were available for export.")

    export_columns = [*feature_names, "target", "partition", "source_udi"]
    export_frame = pd.concat(transformed, axis=0, ignore_index=True)
    export_frame = export_frame.loc[:, export_columns]

    split_manifest = build_split_manifest(splits)
    manifest_bytes = _canonical_json(split_manifest).encode("utf-8")
    metadata: dict[str, object] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "raw_source": str(RAW_DATA_RELATIVE_PATH).replace("\\", "/"),
        "raw_dvc_hash": dvc_hash,
        "split_policy": {
            "version": INITIAL_SPLIT_VERSION,
            "random_state": RANDOM_STATE,
            "partitions": ["train", "validation", "future", "test"],
        },
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "partition_counts": partition_counts,
        "row_count": len(export_frame),
        "feature_names": list(feature_names),
        "output_columns": export_columns,
        "source_row_id": {
            "column": "source_udi",
            "meaning": "raw dataset UDI value, not the loaded DataFrame index",
        },
        "preprocessing": {
            "implementation": "DataProcessor.create_preprocessing_pipeline",
            "fit_partition": "train",
            "fit_scope": "train-only; validation, future and test use transform",
            "classifier_trained": False,
        },
        "versions": _package_versions(),
    }

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / EXPORT_CSV_NAME
    metadata_path = output / EXPORT_METADATA_NAME
    _write_csv_atomically(export_frame, csv_path)
    _write_json_atomically(metadata, metadata_path)

    return SubmissionExportResult(
        csv_path=csv_path,
        metadata_path=metadata_path,
        row_count=len(export_frame),
        partition_counts=partition_counts,
        feature_names=feature_names,
        metadata=metadata,
    )


def _partitions(
    splits: DataSplits,
) -> tuple[tuple[str, pd.DataFrame, pd.Series], ...]:
    """Return initial partitions in the stable export order."""
    return (
        ("train", splits.X_train, splits.y_train),
        ("validation", splits.X_validation, splits.y_validation),
        ("future", splits.X_future, splits.y_future),
        ("test", splits.X_test, splits.y_test),
    )


def _canonical_json(value: object) -> str:
    """Serialize metadata deterministically for identifiers and files."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _package_versions() -> dict[str, str]:
    """Return the interpreter and preprocessing dependency versions."""
    versions = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    versions["python"] = platform.python_version()
    versions["scikit-learn-runtime"] = sklearn.__version__
    return versions


def _write_csv_atomically(frame: pd.DataFrame, destination: Path) -> None:
    """Write a deterministic CSV through a temporary sibling file."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_csv(
            temporary,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomically(metadata: dict[str, object], destination: Path) -> None:
    """Write JSON metadata through a temporary sibling file."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
