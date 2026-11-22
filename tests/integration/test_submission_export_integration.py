"""Integration tests for the persisted submission export."""

from pathlib import Path

import pandas as pd
import numpy as np

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.provenance import read_dvc_directory_hash
from predictive_maintenance.data.submission_export import (
    export_preprocessed_submission,
)
from predictive_maintenance.data.split_manifest import build_split_manifest
from predictive_maintenance.data.splitting import create_initial_data_splits
from predictive_maintenance.models.dvc_lineage import INITIAL_SPLIT_VERSION


def test_export_preserves_raw_rows_and_initial_split_membership(
    tmp_path: Path,
) -> None:
    """The real export matches raw identifiers, targets, and the seed-42 split."""
    repository_root = Path(__file__).resolve().parents[2]
    raw = load_raw_data(repository_root / "data/raw/predictive_maintenance_5000.csv")
    processor = DataProcessor()
    normalized = processor.rename_columns(raw)
    features, target = processor.split_features_target(normalized)
    splits = create_initial_data_splits(features, target)

    result = export_preprocessed_submission(repository_root, tmp_path / "submission")
    exported = pd.read_csv(result.csv_path)
    metadata = result.metadata

    expected_features = list(result.feature_names)
    assert expected_features == [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temperature_difference_k",
        "type_H",
        "type_L",
        "type_M",
    ]
    assert list(exported.columns) == [
        *expected_features,
        "target",
        "partition",
        "source_udi",
    ]
    assert len(exported) == 5000
    assert set(exported["partition"]) == {"train", "validation", "future", "test"}
    assert exported[expected_features].select_dtypes(exclude="number").empty
    assert exported[expected_features].notna().all().all()
    assert np.isfinite(exported[expected_features].to_numpy()).all()
    assert exported["target"].isin([0, 1]).all()
    assert exported["source_udi"].is_unique
    assert set(exported["source_udi"]) == set(raw["UDI"])

    expected_by_udi: dict[int, tuple[str, int]] = {}
    for name in ("train", "validation", "future", "test"):
        partition_target = getattr(splits, f"y_{name}")
        for index, value in partition_target.items():
            expected_by_udi[int(raw.loc[index, "UDI"])] = (name, int(value))

    actual_by_udi = {
        int(row.source_udi): (row.partition, int(row.target))
        for row in exported.itertuples(index=False)
    }
    assert actual_by_udi == expected_by_udi
    assert len(actual_by_udi) == 5000

    partition_sets = {
        name: set(exported.loc[exported["partition"] == name, "source_udi"])
        for name in ("train", "validation", "future", "test")
    }
    for index, first in enumerate(partition_sets.values()):
        for second in list(partition_sets.values())[index + 1:]:
            assert first.isdisjoint(second)

    manifest = build_split_manifest(splits)
    assert metadata["partition_counts"] == {
        "train": 3000,
        "validation": 750,
        "future": 750,
        "test": 500,
    }
    assert metadata["feature_names"] == expected_features
    assert metadata["raw_dvc_hash"] == read_dvc_directory_hash(repository_root)
    assert metadata["split_policy"]["version"] == INITIAL_SPLIT_VERSION
    assert metadata["split_manifest_sha256"]
    assert manifest["row_index_meaning"] == "index of the loaded source dataset"


def test_non_train_changes_do_not_change_fitted_train_transform() -> None:
    """Changing validation rows cannot update train-fitted preprocessing state."""
    repository_root = Path(__file__).resolve().parents[2]
    raw = load_raw_data(repository_root / "data/raw/predictive_maintenance_5000.csv")
    processor = DataProcessor()
    normalized = processor.rename_columns(raw)
    features, target = processor.split_features_target(normalized)
    splits = create_initial_data_splits(features, target)

    pipeline = processor.create_preprocessing_pipeline()
    train_before = pipeline.fit_transform(splits.X_train)
    scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    mean_before = scaler.mean_.copy()

    changed_validation = splits.X_validation.copy()
    changed_validation["air_temperature_k"] = 10_000.0
    changed_validation["process_temperature_k"] = 20_000.0
    pipeline.transform(changed_validation)

    train_after = pipeline.transform(splits.X_train)
    pd.testing.assert_frame_equal(train_before, train_after)
    assert np.array_equal(mean_before, scaler.mean_)
