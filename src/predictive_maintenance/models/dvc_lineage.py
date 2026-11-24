"""Connect verified Git and DVC references to MLflow dataset lineage."""

from pathlib import Path

from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    verify_dvc_dataset,
)
from predictive_maintenance.data.splitting import (
    FINAL_TEST_SIZE,
    FUTURE_SIZE,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)
from predictive_maintenance.models.data_lineage import DatasetLineage


INITIAL_SPLIT_VERSION = (
    f"stratified-v1-seed{RANDOM_STATE}"
    f"-train{TRAIN_SIZE}-validation{VALIDATION_SIZE}"
    f"-future{FUTURE_SIZE}-test{FINAL_TEST_SIZE}"
)


def build_initial_dataset_lineage(
    repository_root: Path,
) -> DatasetLineage:
    """Resolve provenance for the default initial split policy."""
    git_commit = read_clean_git_commit(repository_root)
    dvc_hash = verify_dvc_dataset(repository_root)

    return DatasetLineage(
        dataset_name="predictive_maintenance_5000",
        dvc_hash=dvc_hash,
        git_commit=git_commit,
        split_version=INITIAL_SPLIT_VERSION,
    )
