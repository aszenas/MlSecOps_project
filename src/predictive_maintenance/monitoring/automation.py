"""Coordinate monitoring, prediction export and optional retraining."""

import sqlite3
from datetime import datetime
from pathlib import Path

from mlflow.exceptions import MlflowException

from predictive_maintenance.monitoring.monitoring_pipeline import run_monitoring
from predictive_maintenance.monitoring.prediction_export import (
    export_prediction_events,
)
from predictive_maintenance.monitoring.retraining import run_retraining
from predictive_maintenance.monitoring.sample_status import (
    DEFAULT_MINIMUM_CURRENT_SAMPLES,
)


CYCLE_ERRORS = (OSError, sqlite3.Error, MlflowException, RuntimeError, ValueError)


def run_monitoring_cycle(
    repository_root: Path,
    database_path: Path,
    start: datetime,
    end: datetime,
    output_dir: Path,
    tracking_uri: str,
    minimum_samples: int = DEFAULT_MINIMUM_CURRENT_SAMPLES,
    *,
    retraining_enabled: bool = False,
) -> dict[str, object]:
    """Publish monitoring, export pending events and optionally retrain."""
    monitoring = run_monitoring(
        repository_root=repository_root,
        database_path=database_path,
        start=start,
        end=end,
        output_dir=output_dir,
        minimum_samples=minimum_samples,
    )
    state = monitoring.to_dict()

    export_result: dict[str, object]
    try:
        export_result = export_prediction_events(
            database_path=database_path,
            tracking_uri=tracking_uri,
        ) or {"status": "no_matching_events"}
    except CYCLE_ERRORS as exc:
        export_result = {"status": "failed", "error": str(exc)}

    retraining_result: dict[str, object] = {"status": "disabled"}
    if retraining_enabled:
        try:
            retraining_result = run_retraining(
                repository_root=repository_root,
                database_path=database_path,
                monitoring_state=state,
                tracking_uri=tracking_uri,
            )
        except CYCLE_ERRORS as exc:
            retraining_result = {"status": "failed", "error": str(exc)}

    failed = (
        export_result.get("status") == "failed"
        or retraining_result.get("status") == "failed"
        or retraining_result.get("audit_status") == "failed"
    )
    return {
        "status": "partial_failure" if failed else "completed",
        "monitoring": state,
        "export": export_result,
        "retraining": retraining_result,
    }
