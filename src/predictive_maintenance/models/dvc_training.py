"""Train registered models using a verified DVC dataset."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    verify_dvc_dataset,
    verify_dvc_training_file,
)
from predictive_maintenance.data.split_manifest import DatasetVersion
from predictive_maintenance.data.splitting import (
    DataSplits,
    RANDOM_STATE,
    create_initial_data_splits,
    create_retraining_data_splits,
)
from predictive_maintenance.models.dvc_lineage import build_initial_dataset_lineage
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.models.training_registry import (
    TrainingRegistryResult,
    train_register_and_select,
)


def _load_v2_training_split(
    repository_root: Path,
    raw_data: pd.DataFrame,
    initial_splits: DataSplits,
    processor: DataProcessor,
) -> tuple[DataSplits, str]:
    """Load the verified v2 CSV while retaining the original split indexes."""
    training_hash = verify_dvc_training_file(repository_root)
    training_rows = load_raw_data(repository_root / "data/processed/training/train_v2.csv")

    source_udis = cast(pd.Series, raw_data["UDI"])
    expected = create_retraining_data_splits(initial_splits, source_udis)
    expected_rows = raw_data.loc[expected.X_train.index].reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            training_rows, expected_rows, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError("The DVC v2 training CSV does not match the expected v2 split.") from exc

    training_rows.index = expected.X_train.index
    features, target = processor.split_features_target(processor.rename_columns(training_rows))

    if verify_dvc_training_file(repository_root) != training_hash:
        raise ValueError("DVC v2 training file changed during preparation.")

    return replace(expected, X_train=features, y_train=target), training_hash


def train_from_dvc(
    repository_root: Path,
    model_type: ModelType = "gradient_boosting",
    tracking_uri: str | None = None,
    *,
    dataset_version: DatasetVersion = "v1",
) -> TrainingRegistryResult:
    """Verify provenance, prepare the selected split and invoke Registry training."""
    if dataset_version not in ("v1", "v2"):
        raise ValueError(f"Unsupported dataset version: {dataset_version}.")

    lineage = build_initial_dataset_lineage(repository_root)
    dataset_path = repository_root / "data/raw/predictive_maintenance_5000.csv"
    raw_data = load_raw_data(dataset_path)

    processor = DataProcessor()
    normalized = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized)
    splits = create_initial_data_splits(
        features,
        target,
        random_state=RANDOM_STATE,
    )

    if dataset_version == "v2":
        splits, training_hash = _load_v2_training_split(repository_root, raw_data, splits, processor)
        lineage = replace(
            lineage,
            split_version=f"stratified-v2-seed{RANDOM_STATE}-future-to-train",
            training_dvc_hash=training_hash,
        )

    if verify_dvc_dataset(repository_root) != lineage.dvc_hash:
        raise ValueError("DVC dataset version changed during preparation.")

    if read_clean_git_commit(repository_root) != lineage.git_commit:
        raise ValueError("Git commit changed during preparation.")

    return train_register_and_select(
        splits=splits,
        model_type=model_type,
        tracking_uri=tracking_uri,
        dataset_lineage=lineage,
        record_split_manifest=True,
        dataset_version=dataset_version,
    )
