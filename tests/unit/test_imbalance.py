
"""Unit tests for class imbalance analysis and sample weighting."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.predictive_maintenance.data.imbalance import (
    calculate_sample_weights,
    create_class_distribution,
    validate_binary_target,
)


@pytest.fixture
def imbalanced_target() -> pd.Series:
    """Create a binary target with a minority failure class."""
    return pd.Series(
        [0] * 90 + [1] * 10,
        name="target",
    )


def test_valid_binary_target(imbalanced_target: pd.Series) -> None:
    """Valid binary targets should pass validation."""
    assert validate_binary_target(imbalanced_target) is None


def test_empty_target() -> None:
    """Empty targets should be rejected."""
    with pytest.raises(ValueError, match="empty"):
        validate_binary_target(pd.Series(dtype=int))


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2],
        [0, 1, -1],
        [0, 1, None],
    ],
)
def test_invalid_target_values(values: list[int | None]) -> None:
    """Invalid target labels should be rejected."""
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_binary_target(pd.Series(values))


def test_single_class_target() -> None:
    """Both binary classes must be present."""
    with pytest.raises(ValueError, match="Both target classes"):
        validate_binary_target(pd.Series([0, 0, 0]))


def test_class_distribution(imbalanced_target: pd.Series) -> None:
    """Class counts and percentages should be correct."""
    result = create_class_distribution(imbalanced_target)

    assert result.loc[0, "count"] == 90
    assert result.loc[1, "count"] == 10
    assert result.loc[0, "ratio_percent"] == 90.0
    assert result.loc[1, "ratio_percent"] == 10.0


def test_sample_weights(imbalanced_target: pd.Series) -> None:
    """Balanced sample weights should match the expected formula."""
    weights = calculate_sample_weights(imbalanced_target)

    normal_weight = cast(pd.Series, weights.loc[imbalanced_target == 0]).iloc[0]
    failure_weight = cast(pd.Series, weights.loc[imbalanced_target == 1]).iloc[0]

    assert np.isclose(normal_weight, 100 / (2 * 90))
    assert np.isclose(failure_weight, 100 / (2 * 10))
    assert failure_weight > normal_weight


def test_weights_preserve_indexes(imbalanced_target: pd.Series) -> None:
    """Sample weights should align with the original training labels."""
    target = imbalanced_target.copy()
    target.index = range(100, 200)

    weights = calculate_sample_weights(target)

    assert weights.index.equals(target.index)
    assert weights.name == "sample_weight"
    assert len(weights) == len(target)


def test_total_weight(imbalanced_target: pd.Series) -> None:
    """Balanced sample weights should sum to the sample count."""
    weights = calculate_sample_weights(imbalanced_target)

    assert np.isclose(weights.sum(), len(imbalanced_target))


def test_retraining_weights() -> None:
    """Class weights should adapt when the training distribution changes."""
    initial = pd.Series([0] * 90 + [1] * 10)
    updated = pd.Series([0] * 90 + [1] * 20)

    initial_weights = calculate_sample_weights(initial)
    updated_weights = calculate_sample_weights(updated)

    initial_failure_weight = cast(pd.Series, initial_weights.loc[initial == 1]).iloc[0]
    updated_failure_weight = cast(pd.Series, updated_weights.loc[updated == 1]).iloc[0]

    assert updated_failure_weight < initial_failure_weight
