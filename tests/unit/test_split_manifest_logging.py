"""Test MLflow persistence of exact dataset partition membership."""

from dataclasses import replace
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from predictive_maintenance.data.split_manifest import build_split_manifest
from predictive_maintenance.data.splitting import (
    DataSplits,
    create_initial_data_splits,
    create_retraining_data_splits,
)
from predictive_maintenance.models.data_lineage import (
    DatasetLineage,
    log_dataset_lineage,
    log_split_manifest,
)


@pytest.fixture
def splits() -> DataSplits:
    features = pd.DataFrame({"feature": range(5000)})
    target = pd.Series([0, 1] * 2500)
    return create_initial_data_splits(features, target)


def test_logs_artifact_before_manifest_identifier(splits: DataSplits) -> None:
    client = MagicMock()

    digest = log_split_manifest(client, "run-123", splits)

    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert client.mock_calls == [
        call.log_dict(
            "run-123",
            build_split_manifest(splits),
            "lineage/split_manifest.json",
        ),
        call.set_tag("run-123", "dataset_split_manifest_sha256", digest),
    ]


def test_artifact_failure_prevents_manifest_tag(splits: DataSplits) -> None:
    client = MagicMock()
    client.log_dict.side_effect = RuntimeError("storage failed")

    with pytest.raises(RuntimeError, match="storage failed"):
        log_split_manifest(client, "run-123", splits)

    client.set_tag.assert_not_called()


def test_invalid_split_prevents_logging(splits: DataSplits) -> None:
    client = MagicMock()
    invalid = replace(splits, X_train=splits.X_train.iloc[:-1])

    with pytest.raises(ValueError, match="partition size"):
        log_split_manifest(client, "run-123", invalid)

    client.log_dict.assert_not_called()
    client.set_tag.assert_not_called()


def test_row_order_changes_manifest_identifier(splits: DataSplits) -> None:
    reordered = replace(
        splits,
        X_train=splits.X_train.iloc[::-1],
        y_train=splits.y_train.iloc[::-1],
    )

    original_digest = log_split_manifest(MagicMock(), "run-1", splits)
    reordered_digest = log_split_manifest(MagicMock(), "run-2", reordered)

    assert original_digest != reordered_digest


def test_logs_v2_partition_membership_and_separate_dvc_hashes(splits: DataSplits) -> None:
    source_udis = pd.Series(range(5000))
    moved = tuple(int(index) for index in splits.X_future.index[:10])
    v2 = create_retraining_data_splits(splits, source_udis, moved)
    client = MagicMock()
    lineage = DatasetLineage("dataset", dvc_hash="raw.dir", training_dvc_hash="train")

    log_dataset_lineage(client, "run-v2", v2, lineage)
    digest = log_split_manifest(client, "run-v2", v2, dataset_version="v2")

    metadata = client.log_dict.call_args_list[0].args[1]
    assert metadata["dvc_hash"] == "raw.dir"
    assert metadata["training_dvc_hash"] == "train"
    assert metadata["split_rows"] == {"train": 3010, "validation": 750, "future": 740, "test": 500}
    client.set_tag.assert_any_call("run-v2", "dataset_training_dvc_hash", "train")
    client.set_tag.assert_any_call("run-v2", "dataset_split_manifest_sha256", digest)

    manifest = client.log_dict.call_args_list[1].args[1]
    assert manifest["row_indices"] == {
        "train": splits.X_train.index.tolist() + list(moved),
        "validation": splits.X_validation.index.tolist(),
        "future": splits.X_future.index[10:].tolist(),
        "test": splits.X_test.index.tolist(),
    }

    with pytest.raises(ValueError, match="partition size"):
        build_split_manifest(v2)
