
"""Unit tests for deterministic feature drift simulation."""

import pandas as pd
import pytest

from predictive_maintenance.monitoring import simulation


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create representative model input features."""
    return pd.DataFrame(
        {
            "type": ["L", "M"],
            "air_temperature_k": [300.0, 301.0],
            "process_temperature_k": [310.0, 311.0],
            "rotational_speed_rpm": [1500, 1600],
            "torque_nm": [40.0, 45.0],
            "tool_wear_min": [100, 120],
        }
    )


@pytest.fixture(autouse=True)
def mock_feature_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate simulation from preprocessing implementation."""

    def select_features(
        self: object,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        return data.copy(deep=True)

    monkeypatch.setattr(
        simulation.DataProcessor,
        "select_features",
        select_features,
    )


def test_simulation_applies_expected_shifts(
    sample_features: pd.DataFrame,
) -> None:
    """Change exactly the four configured numeric features."""
    result = simulation.simulate_feature_drift(sample_features)

    expected = sample_features.copy(deep=True)
    expected["air_temperature_k"] += 10.0
    expected["process_temperature_k"] += 10.0
    expected["rotational_speed_rpm"] *= 1.4
    expected["torque_nm"] += 20.0

    pd.testing.assert_frame_equal(result, expected)


def test_simulation_preserves_input(
    sample_features: pd.DataFrame,
) -> None:
    """Do not modify the caller's original DataFrame."""
    original = sample_features.copy(deep=True)

    simulation.simulate_feature_drift(sample_features)

    pd.testing.assert_frame_equal(sample_features, original)


def test_simulation_is_deterministic(
    sample_features: pd.DataFrame,
) -> None:
    """Identical inputs must produce identical outputs."""
    first = simulation.simulate_feature_drift(sample_features)
    second = simulation.simulate_feature_drift(sample_features)

    pd.testing.assert_frame_equal(first, second)


def test_simulation_preserves_shape_and_columns(
    sample_features: pd.DataFrame,
) -> None:
    """Keep the input feature schema and row count."""
    result = simulation.simulate_feature_drift(sample_features)

    assert result.shape == sample_features.shape
    assert result.columns.tolist() == sample_features.columns.tolist()
    assert result.index.equals(sample_features.index)
