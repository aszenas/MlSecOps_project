"""Decide whether monitoring has enough valid current samples."""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal


DEFAULT_MINIMUM_CURRENT_SAMPLES = 100
"""Documented project parameter for the minimum current sample count."""

MonitoringSampleStatus = Literal["insufficient_data", "ready"]


@dataclass(frozen=True)
class MonitoringSampleResult:
    """JSON-friendly result of the valid current sample sufficiency check."""

    status: MonitoringSampleStatus
    actual_count: int
    required_count: int
    missing_count: int

    def to_dict(self) -> dict[str, str | int]:
        """Return the result as a JSON-serializable mapping."""
        return {
            "status": self.status,
            "actual_count": self.actual_count,
            "required_count": self.required_count,
            "missing_count": self.missing_count,
        }


def _validate_positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _validate_current_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("current_count must be a non-negative integer.")
    return int(value)


def assess_monitoring_sample_count(
    current_count: int,
    minimum_samples: int = DEFAULT_MINIMUM_CURRENT_SAMPLES,
) -> MonitoringSampleResult:
    """Assess the number of valid current feature rows for monitoring.

    ``current_count`` must be the count after feature validation. The default
    threshold of 100 is a documented project parameter, not a statistical guarantee.
    A count equal to the threshold is ready for processing. This function only
    evaluates supplied counts and does not access data sources or generate reports.
    """
    actual_count = _validate_current_count(current_count)
    required_count = _validate_positive_integer(minimum_samples, "minimum_samples")
    missing_count = max(required_count - actual_count, 0)
    status: MonitoringSampleStatus = (
        "insufficient_data" if missing_count > 0 else "ready"
    )
    return MonitoringSampleResult(
        status=status,
        actual_count=actual_count,
        required_count=required_count,
        missing_count=missing_count,
    )
