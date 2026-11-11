"""Unit tests for dataset lineage records attached to MLflow runs."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictive_maintenance.models.data_lineage import (
    DatasetLineage,
    log_dataset_lineage,
)


@pytest.fixture
def fake_splits() -> MagicMock:
    """Provide split frames without running preprocessing."""
    splits = MagicMock()
    splits.X_train = pd.DataFrame({"x": [1, 2, 3]})
    splits.X_validation = pd.DataFrame({"x": [4, 5]})
    splits.X_future = pd.DataFrame({"x": [6]})
    splits.X_test = pd.DataFrame({"x": [7, 8, 9, 10]})
    return splits


def test_logs_lineage_artifact_and_searchable_tags(
    fake_splits: MagicMock,
) -> None:
    """Persist exact supplied version IDs and actual split counts."""
    client = MagicMock()
    lineage = DatasetLineage(
        dataset_name="predictive_maintenance_5000",
        dvc_hash="abcdef123456",
        git_commit="123456abcdef",
        split_version="initial-seed-42",
    )

    log_dataset_lineage(client, "run-123", fake_splits, lineage)

    client.log_dict.assert_called_once_with(
        "run-123",
        {
            "dataset_name": "predictive_maintenance_5000",
            "dvc_hash": "abcdef123456",
            "git_commit": "123456abcdef",
            "split_version": "initial-seed-42",
            "split_rows": {
                "train": 3,
                "validation": 2,
                "future": 1,
                "test": 4,
            },
        },
        "lineage/dataset_metadata.json",
    )
    client.set_tag.assert_any_call(
        "run-123", "dataset_dvc_hash", "abcdef123456"
    )
    client.set_tag.assert_any_call(
        "run-123", "dataset_git_commit", "123456abcdef"
    )
    assert client.set_tag.call_count == 4


def test_optional_version_ids_are_not_invented(
    fake_splits: MagicMock,
) -> None:
    """An unversioned dataset must not acquire fabricated hashes."""
    client = MagicMock()

    log_dataset_lineage(
        client, "run-123", fake_splits, DatasetLineage("dataset")
    )

    client.set_tag.assert_called_once_with(
        "run-123", "dataset_name", "dataset"
    )


@pytest.mark.parametrize(
    "lineage",
    [
        DatasetLineage(" "),
        DatasetLineage("dataset", dvc_hash=" "),
        DatasetLineage("dataset", git_commit=""),
        DatasetLineage("dataset", split_version=" "),
    ],
)
def test_rejects_empty_identifiers_before_logging(
    fake_splits: MagicMock,
    lineage: DatasetLineage,
) -> None:
    """Do not log incomplete identifiers as legitimate lineage."""
    client = MagicMock()

    with pytest.raises(ValueError):
        log_dataset_lineage(client, "run-123", fake_splits, lineage)

    client.log_dict.assert_not_called()
    client.set_tag.assert_not_called()


def test_artifact_failure_prevents_tags(fake_splits: MagicMock) -> None:
    """Do not write searchable tags when the artifact was not saved."""
    client = MagicMock()
    client.log_dict.side_effect = RuntimeError("artifact failed")

    with pytest.raises(RuntimeError, match="artifact failed"):
        log_dataset_lineage(
            client, "run-123", fake_splits, DatasetLineage("dataset")
        )

    client.set_tag.assert_not_called()
