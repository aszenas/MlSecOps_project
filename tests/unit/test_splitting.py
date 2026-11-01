
"""Unit tests for reproducible initial dataset splitting."""

import numpy as np
import pandas as pd
import pytest

from src.predictive_maintenance.data.splitting import (
    create_initial_data_splits,
)


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create 5000 samples with imbalanced binary targets."""
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        {
            "air_temperature_k": rng.normal(300, 2, 5000),
            "torque_nm": rng.normal(40, 10, 5000),
        }
    )

    y = pd.Series(
        [0] * 4837 + [1] * 163,
        index=X.index,
        name="target",
    )

    return X, y


def test_split_sizes(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify all four partition sizes."""
    X, y = sample_dataset

    splits = create_initial_data_splits(X, y)

    assert len(splits.X_train) == 3000
    assert len(splits.X_validation) == 750
    assert len(splits.X_future) == 750
    assert len(splits.X_test) == 500

    assert len(splits.y_train) == 3000
    assert len(splits.y_validation) == 750
    assert len(splits.y_future) == 750
    assert len(splits.y_test) == 500


def test_no_overlap(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Ensure partitions share no original record indexes."""
    X, y = sample_dataset

    splits = create_initial_data_splits(X, y)

    partitions = [
        set(splits.X_train.index),
        set(splits.X_validation.index),
        set(splits.X_future.index),
        set(splits.X_test.index),
    ]

    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            assert partitions[i].isdisjoint(partitions[j])


def test_all_records_preserved(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Ensure every input record appears in exactly one partition."""
    X, y = sample_dataset

    splits = create_initial_data_splits(X, y)

    all_indices = set().union(
        splits.X_train.index,
        splits.X_validation.index,
        splits.X_future.index,
        splits.X_test.index,
    )

    assert all_indices == set(X.index)


def test_reproducibility(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Same random seed should produce identical partitions."""
    X, y = sample_dataset

    first = create_initial_data_splits(X, y, random_state=42)
    second = create_initial_data_splits(X, y, random_state=42)

    for name in ("train", "validation", "future", "test"):
        pd.testing.assert_frame_equal(
            getattr(first, f"X_{name}"),
            getattr(second, f"X_{name}"),
        )

        pd.testing.assert_series_equal(
            getattr(first, f"y_{name}"),
            getattr(second, f"y_{name}"),
        )


def test_stratification(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Target proportions should remain close across partitions."""
    X, y = sample_dataset

    splits = create_initial_data_splits(X, y)

    original_rate = y.mean()

    for target in (
        splits.y_train,
        splits.y_validation,
        splits.y_future,
        splits.y_test,
    ):
        assert abs(target.mean() - original_rate) < 0.01


def test_feature_target_alignment(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Features and labels must remain aligned after splitting."""
    X, y = sample_dataset

    splits = create_initial_data_splits(X, y)

    for name in ("train", "validation", "future", "test"):
        features = getattr(splits, f"X_{name}")
        target = getattr(splits, f"y_{name}")

        assert features.index.equals(target.index)
        pd.testing.assert_series_equal(
            target,
            y.loc[features.index],
        )


def test_invalid_dataset_size(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Unexpected initial dataset size should be rejected."""
    X, y = sample_dataset

    with pytest.raises(ValueError, match="Expected 5000 records"):
        create_initial_data_splits(X.iloc[:100], y.iloc[:100])


def test_misaligned_indexes(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Features and labels with different indexes should be rejected."""
    X, y = sample_dataset
    invalid_y = y.copy()

    invalid_y.index = invalid_y.index + 1

    with pytest.raises(ValueError, match="identical indexes"):
        create_initial_data_splits(X, invalid_y)


def test_duplicate_indexes(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Duplicate record indexes should be rejected."""
    X, y = sample_dataset

    invalid_X = X.copy()
    invalid_y = y.copy()

    new_index = list(range(4999)) + [0]
    invalid_X.index = new_index
    invalid_y.index = new_index

    with pytest.raises(ValueError, match="indexes must be unique"):
        create_initial_data_splits(invalid_X, invalid_y)


def test_single_target_class(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Splitting requires both target classes."""
    X, y = sample_dataset

    invalid_y = pd.Series(0, index=y.index)

    with pytest.raises(ValueError, match="Both target classes"):
        create_initial_data_splits(X, invalid_y)


@pytest.mark.parametrize("invalid_value", [2, -1, None])
def test_invalid_target_values(
    sample_dataset: tuple[pd.DataFrame, pd.Series],
    invalid_value: int | None,
) -> None:
    """Invalid or missing target values should be rejected."""
    X, y = sample_dataset
    invalid_y = y.copy()
    invalid_y.iloc[0] = invalid_value

    with pytest.raises(ValueError, match="Target must contain"):
        create_initial_data_splits(X, invalid_y)
