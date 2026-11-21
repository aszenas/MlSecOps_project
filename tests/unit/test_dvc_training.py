"""Verify that invalid provenance prevents Registry training."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from predictive_maintenance.models.data_lineage import DatasetLineage
from predictive_maintenance.models import dvc_training


COMMIT = "a" * 40
DVC_HASH = "b" * 32 + ".dir"


@pytest.fixture
def dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    names = [
        "build_initial_dataset_lineage",
        "load_raw_data",
        "DataProcessor",
        "create_initial_data_splits",
        "verify_dvc_dataset",
        "read_clean_git_commit",
        "train_register_and_select",
    ]
    mocks = {name: MagicMock() for name in names}
    for name, mock in mocks.items():
        monkeypatch.setattr(dvc_training, name, mock)

    mocks["build_initial_dataset_lineage"].return_value = DatasetLineage(
        dataset_name="predictive_maintenance_5000",
        dvc_hash=DVC_HASH,
        git_commit=COMMIT,
        split_version="initial-policy",
    )
    processor = mocks["DataProcessor"].return_value
    processor.split_features_target.return_value = (
        MagicMock(),
        MagicMock(),
    )
    mocks["verify_dvc_dataset"].return_value = DVC_HASH
    mocks["read_clean_git_commit"].return_value = COMMIT
    return mocks


def test_passes_verified_lineage_to_training(
    tmp_path: Path,
    dependencies: dict[str, MagicMock],
) -> None:
    result = dvc_training.train_from_dvc(
        tmp_path,
        model_type="xgboost",
        tracking_uri="sqlite:///tracking.db",
    )

    train = dependencies["train_register_and_select"]
    assert result is train.return_value
    train.assert_called_once_with(
        splits=dependencies["create_initial_data_splits"].return_value,
        model_type="xgboost",
        tracking_uri="sqlite:///tracking.db",
        dataset_lineage=dependencies[
            "build_initial_dataset_lineage"
        ].return_value,
        record_split_manifest=True,
    )
    assert dependencies[
        "create_initial_data_splits"
    ].call_args.kwargs["random_state"] == 42


@pytest.mark.parametrize(
    "dependency",
    [
        "build_initial_dataset_lineage",
        "load_raw_data",
        "verify_dvc_dataset",
        "read_clean_git_commit",
    ],
)
def test_preparation_failure_prevents_training(
    tmp_path: Path,
    dependencies: dict[str, MagicMock],
    dependency: str,
) -> None:
    dependencies[dependency].side_effect = ValueError("verification failed")

    with pytest.raises(ValueError, match="verification failed"):
        dvc_training.train_from_dvc(tmp_path)

    dependencies["train_register_and_select"].assert_not_called()


@pytest.mark.parametrize(
    ("dependency", "new_value", "message"),
    [
        ("verify_dvc_dataset", "c" * 32 + ".dir", "DVC dataset version changed"),
        ("read_clean_git_commit", "c" * 40, "Git commit changed"),
    ],
)
def test_version_change_prevents_training(
    tmp_path: Path,
    dependencies: dict[str, MagicMock],
    dependency: str,
    new_value: str,
    message: str,
) -> None:
    dependencies[dependency].return_value = new_value

    with pytest.raises(ValueError, match=message):
        dvc_training.train_from_dvc(tmp_path)

    dependencies["train_register_and_select"].assert_not_called()
