"""Coordinate monitoring and prediction export without starting training."""

import sqlite3
from datetime import datetime
from pathlib import Path

from mlflow.exceptions import MlflowException

from predictive_maintenance.monitoring.monitoring_pipeline import run_monitoring
from predictive_maintenance.monitoring.prediction_export import (
    export_prediction_events,
)
from predictive_maintenance.monitoring.sample_status import (
    DEFAULT_MINIMUM_CURRENT_SAMPLES,
)


def run_monitoring_cycle(
    repository_root: Path,
    database_path: Path,
    start: datetime,
    end: datetime,
    output_dir: Path,
    tracking_uri: str,
    minimum_samples: int = DEFAULT_MINIMUM_CURRENT_SAMPLES,
) -> dict[str, object]:
    """Publish monitoring results and export pending prediction events."""
    monitoring = run_monitoring(
        repository_root=repository_root,
        database_path=database_path,
        start=start,
        end=end,
        output_dir=output_dir,
        minimum_samples=minimum_samples,
    )

    try:
        exported = export_prediction_events(
            database_path=database_path,
            tracking_uri=tracking_uri,
        )
    except (OSError, sqlite3.Error, MlflowException, RuntimeError, ValueError) as exc:
        return {
            "status": "partial_failure",
            "monitoring": monitoring.to_dict(),
            "export": {"status": "failed", "error": str(exc)},
        }

    return {
        "status": "completed",
        "monitoring": monitoring.to_dict(),
        "export": exported or {"status": "no_matching_events"},
    }
