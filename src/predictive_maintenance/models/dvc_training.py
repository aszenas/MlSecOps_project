"""Train registered models using a verified DVC dataset."""

from pathlib import Path

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    verify_dvc_dataset,
)
from predictive_maintenance.data.splitting import (
    RANDOM_STATE,
    create_initial_data_splits,
)
from predictive_maintenance.models.dvc_lineage import (
    build_initial_dataset_lineage,
)
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.models.training_registry import (
    TrainingRegistryResult,
    train_register_and_select,
)


def train_from_dvc(
    repository_root: Path,
    model_type: ModelType = "gradient_boosting",
    tracking_uri: str | None = None,
) -> TrainingRegistryResult:
    """Verify provenance, prepare initial splits and invoke Registry training."""
    lineage = build_initial_dataset_lineage(repository_root)
    dataset_path = (
        repository_root / "data" / "raw" / "predictive_maintenance_5000.csv"
    )
    raw_data = load_raw_data(dataset_path)

    processor = DataProcessor()
    normalized = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized)
    splits = create_initial_data_splits(
        features,
        target,
        random_state=RANDOM_STATE,
    )

    # Recheck after loading and splitting, before training side effects.
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
    )
