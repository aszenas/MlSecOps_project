"""Decide whether monitoring results allow retraining."""

from collections.abc import Mapping
from typing import Literal


RetrainingDecision = Literal[
    "insufficient_data",
    "no_drift",
    "waiting_for_labeled_data",
    "eligible",
]


def decide_retraining(
    monitoring_state: Mapping[str, object],
    *,
    labeled_data_available: bool = False,
) -> RetrainingDecision:
    """Require a valid drift result and verified labeled training data."""
    status = monitoring_state.get("status")

    if status == "insufficient_data":
        return "insufficient_data"

    if status != "ready":
        raise ValueError("Retraining requires a valid monitoring status.")

    drift = monitoring_state.get("drift_decision")
    if not isinstance(drift, bool):
        raise ValueError("Ready monitoring must contain a boolean drift decision.")

    if not drift:
        return "no_drift"

    if not labeled_data_available:
        return "waiting_for_labeled_data"

    return "eligible"
