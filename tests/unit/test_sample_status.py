"""Unit tests for monitoring sample sufficiency."""

import json

import pytest

from predictive_maintenance.monitoring.sample_status import (
    assess_monitoring_sample_count,
)


@pytest.mark.parametrize(
    "current_count, status, missing",
    [(0, "insufficient_data", 100), (99, "insufficient_data", 1), (100, "ready", 0), (101, "ready", 0)],
)
def test_sample_boundaries(current_count: int, status: str, missing: int) -> None:
    """Treat exactly the required count as ready."""
    result = assess_monitoring_sample_count(current_count)

    assert result.status == status
    assert result.actual_count == current_count
    assert result.required_count == 100
    assert result.missing_count == missing


def test_supports_custom_threshold_and_json_serialization() -> None:
    """Return a JSON-friendly result for a configured threshold."""
    result = assess_monitoring_sample_count(7, minimum_samples=8)

    assert json.loads(json.dumps(result.to_dict())) == {
        "status": "insufficient_data",
        "actual_count": 7,
        "required_count": 8,
        "missing_count": 1,
    }


@pytest.mark.parametrize("minimum", [True, False, 0, -1, 1.5, "100", None])
def test_rejects_invalid_minimum(minimum: object) -> None:
    """Require a positive non-boolean integer threshold."""
    with pytest.raises(ValueError, match="positive integer"):
        assess_monitoring_sample_count(0, minimum_samples=minimum)  # type: ignore[arg-type]


@pytest.mark.parametrize("current", [True, -1, 1.5, "1", None])
def test_rejects_invalid_current_count(current: object) -> None:
    """Reject invalid current counts while allowing zero."""
    with pytest.raises(ValueError, match="non-negative integer"):
        assess_monitoring_sample_count(current)  # type: ignore[arg-type]
