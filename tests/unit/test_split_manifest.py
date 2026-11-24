"""Verify exact split membership and reject invalid partitions."""

from dataclasses import replace
import json

import pandas as pd
import pytest

from predictive_maintenance.data.split_manifest import build_split_manifest
from predictive_maintenance.data.splitting import DataSplits


@pytest.fixture
def splits() -> DataSplits:
    frames = []
    labels = []
    start = 0
    for size in (3000, 750, 750, 500):
        index = pd.RangeIndex(start, start + size)
        frames.append(pd.DataFrame({"feature": 0}, index=index))
        labels.append(pd.Series(0, index=index))
        start += size

    return DataSplits(
        X_train=frames[0],
        X_validation=frames[1],
        X_future=frames[2],
        X_test=frames[3],
        y_train=labels[0],
        y_validation=labels[1],
        y_future=labels[2],
        y_test=labels[3],
    )


def test_preserves_exact_partition_membership(splits: DataSplits) -> None:
    manifest = build_split_manifest(splits)

    assert manifest["row_indices"] == {
        "train": list(range(3000)),
        "validation": list(range(3000, 3750)),
        "future": list(range(3750, 4500)),
        "test": list(range(4500, 5000)),
    }
    assert manifest["random_state"] == 42
    assert json.loads(json.dumps(manifest)) == manifest


def test_preserves_row_order(splits: DataSplits) -> None:
    reordered = replace(
        splits,
        X_train=splits.X_train.iloc[::-1],
        y_train=splits.y_train.iloc[::-1],
    )

    assert build_split_manifest(reordered) != build_split_manifest(splits)


def test_rejects_wrong_partition_size(splits: DataSplits) -> None:
    invalid = replace(splits, X_train=splits.X_train.iloc[:-1])

    with pytest.raises(ValueError, match="partition size"):
        build_split_manifest(invalid)


def test_rejects_misaligned_target(splits: DataSplits) -> None:
    invalid = replace(splits, y_train=splits.y_train.iloc[::-1])

    with pytest.raises(ValueError, match="indexes differ"):
        build_split_manifest(invalid)


@pytest.mark.parametrize("problem", ["duplicate", "overlap", "string"])
def test_rejects_invalid_row_indexes(
    splits: DataSplits,
    problem: str,
) -> None:
    features = splits.X_validation.copy()
    target = splits.y_validation.copy()
    indices = features.index.tolist()
    if problem == "duplicate":
        indices[1] = indices[0]
        index = pd.Index(indices)
    elif problem == "overlap":
        indices[0] = 0
        index = pd.Index(indices)
    else:
        index = features.index.astype(str)

    features.index = index
    target.index = index
    invalid = replace(splits, X_validation=features, y_validation=target)

    with pytest.raises(ValueError):
        build_split_manifest(invalid)
