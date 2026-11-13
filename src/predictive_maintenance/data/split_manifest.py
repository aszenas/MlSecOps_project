"""Describe the exact row membership of the initial dataset splits."""

import sklearn

from predictive_maintenance.data.splitting import (
    DataSplits,
    FINAL_TEST_SIZE,
    FUTURE_SIZE,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)


def build_split_manifest(splits: DataSplits) -> dict[str, object]:
    """Validate split membership and return a JSON-compatible manifest."""
    partitions = {
        "train": (splits.X_train, splits.y_train, TRAIN_SIZE),
        "validation": (
            splits.X_validation,
            splits.y_validation,
            VALIDATION_SIZE,
        ),
        "future": (splits.X_future, splits.y_future, FUTURE_SIZE),
        "test": (splits.X_test, splits.y_test, FINAL_TEST_SIZE),
    }
    row_indices: dict[str, list[int]] = {}
    seen: set[int] = set()

    for name, (features, target, expected_size) in partitions.items():
        if len(features) != expected_size or len(target) != expected_size:
            raise ValueError(f"Unexpected partition size: {name}.")

        if not features.index.equals(target.index):
            raise ValueError(f"Feature and target indexes differ: {name}.")

        indices = features.index.tolist()
        if not all(type(index) is int for index in indices):
            raise ValueError(f"Expected integer row indexes: {name}.")

        rows = [int(index) for index in indices]
        if len(set(rows)) != len(rows):
            raise ValueError(f"Duplicate row indexes: {name}.")

        if seen.intersection(rows):
            raise ValueError(f"Partitions overlap: {name}.")

        seen.update(rows)
        row_indices[name] = rows

    return {
        "schema_version": 1,
        "split_method": "stratified-train-test-split",
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "row_index_meaning": "index of the loaded source dataset",
        "row_indices": row_indices,
    }
