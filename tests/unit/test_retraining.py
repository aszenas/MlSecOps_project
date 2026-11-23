"""Test retraining eligibility without training models."""

import pytest

from predictive_maintenance.monitoring.retraining import decide_retraining


@pytest.mark.parametrize(
    ("status", "drift", "available", "expected"),
    [
        ("insufficient_data", None, False, "insufficient_data"),
        ("insufficient_data", None, True, "insufficient_data"),
        ("ready", False, False, "no_drift"),
        ("ready", False, True, "no_drift"),
        ("ready", True, False, "waiting_for_labeled_data"),
        ("ready", True, True, "eligible"),
    ],
)
def test_retraining_requires_drift_and_labeled_data(
    status: str,
    drift: bool | None,
    available: bool,
    expected: str,
) -> None:
    state: dict[str, object] = {
        "status": status,
        "drift_decision": drift,
    }

    assert decide_retraining(
        state,
        labeled_data_available=available,
    ) == expected


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"status": "unknown"},
        {"status": "ready", "drift_decision": None},
        {"status": "ready", "drift_decision": 1},
    ],
)
def test_invalid_monitoring_cannot_enable_retraining(
    state: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        decide_retraining(state, labeled_data_available=True)
