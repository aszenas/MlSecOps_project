"""Test monitoring-cycle coordination without external services."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from predictive_maintenance.monitoring import automation


START = datetime(2026, 11, 23, tzinfo=timezone.utc)
END = START + timedelta(hours=1)


@pytest.mark.parametrize("monitoring_status", ["ready", "insufficient_data"])
@pytest.mark.parametrize("export_status", ["exported", "empty", "failed"])
def test_cycle_preserves_monitoring_and_reports_export_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    monitoring_status: str,
    export_status: str,
) -> None:
    state = {"status": monitoring_status}
    monitoring_result = MagicMock()
    monitoring_result.to_dict.return_value = state
    monitor = MagicMock(return_value=monitoring_result)
    exporter = MagicMock()

    expected_export: dict[str, object]
    if export_status == "failed":
        exporter.side_effect = RuntimeError("MLflow unavailable.")
        expected_export = {
            "status": "failed",
            "error": "MLflow unavailable.",
        }
    elif export_status == "empty":
        exporter.return_value = None
        expected_export = {"status": "no_matching_events"}
    else:
        expected_export = {"status": "exported", "run_id": "export-run"}
        exporter.return_value = expected_export

    monkeypatch.setattr(automation, "run_monitoring", monitor)
    monkeypatch.setattr(automation, "export_prediction_events", exporter)

    database = tmp_path / "predictions.db"
    reports = tmp_path / "reports"
    result = automation.run_monitoring_cycle(
        repository_root=tmp_path,
        database_path=database,
        start=START,
        end=END,
        output_dir=reports,
        tracking_uri="http://mlflow.test:5000",
        minimum_samples=150,
    )

    assert result == {
        "status": "partial_failure" if export_status == "failed" else "completed",
        "monitoring": state,
        "export": expected_export,
    }
    monitor.assert_called_once_with(
        repository_root=tmp_path,
        database_path=database,
        start=START,
        end=END,
        output_dir=reports,
        minimum_samples=150,
    )
    exporter.assert_called_once_with(
        database_path=database,
        tracking_uri="http://mlflow.test:5000",
    )


def test_monitoring_failure_stops_the_cycle_before_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monitor = MagicMock(side_effect=RuntimeError("Reference unavailable."))
    exporter = MagicMock()
    monkeypatch.setattr(automation, "run_monitoring", monitor)
    monkeypatch.setattr(automation, "export_prediction_events", exporter)

    with pytest.raises(RuntimeError, match="Reference unavailable"):
        automation.run_monitoring_cycle(
            repository_root=tmp_path,
            database_path=tmp_path / "predictions.db",
            start=START,
            end=END,
            output_dir=tmp_path / "reports",
            tracking_uri="http://mlflow.test:5000",
        )

    exporter.assert_not_called()
