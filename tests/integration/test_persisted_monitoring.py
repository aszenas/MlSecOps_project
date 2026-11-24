"""Exercise monitoring from persisted prediction events end to end."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from predictive_maintenance.monitoring.monitoring_pipeline import run_monitoring
from predictive_maintenance.monitoring.monitoring_reference import (
    MonitoringReferenceResult,
    load_monitoring_reference,
)
from predictive_maintenance.monitoring.prediction_store import save_prediction_event
from predictive_maintenance.monitoring.simulation import simulate_feature_drift


WINDOW_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2025, 1, 2, tzinfo=timezone.utc)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def monitoring_reference() -> MonitoringReferenceResult:
    """Use the repository's verified, versioned reference dataset."""
    return load_monitoring_reference(REPOSITORY_ROOT)


def _python_value(value: object) -> object:
    item = getattr(value, "item", None)
    return cast(Callable[[], object], item)() if callable(item) else value


def _save_feature_events(
    database_path: Path,
    features: pd.DataFrame,
    request_prefix: str,
    *,
    status: str = "success",
    start: datetime = WINDOW_START,
) -> None:
    for index, (_, row) in enumerate(features.iterrows()):
        timestamp = start + timedelta(seconds=index + 1)
        inputs = {column: _python_value(row[column]) for column in features.columns}
        save_prediction_event(
            request_id=f"{request_prefix}-{index}",
            event={
                "status": status,
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "inputs": inputs,
            },
            database_path=database_path,
        )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_monitoring_pipeline_publishes_drift_free_report_for_reference_sample(
    tmp_path: Path,
    monitoring_reference: MonitoringReferenceResult,
) -> None:
    """Persisted current data equal to the reference produces a ready report."""
    database_path = tmp_path / "normal" / "predictions.sqlite"
    output_dir = tmp_path / "normal" / "artifacts"
    _save_feature_events(database_path, monitoring_reference.dataframe, "normal")

    result = run_monitoring(
        repository_root=REPOSITORY_ROOT,
        database_path=database_path,
        start=WINDOW_START,
        end=WINDOW_END,
        output_dir=output_dir,
        minimum_samples=100,
    )

    assert result.status == "ready"
    assert result.state["selected_event_count"] == 3000
    assert result.state["valid_current_count"] == 3000
    assert result.state["skipped_invalid_records"] == 0
    assert result.state["skipped_invalid_features"] == 0
    assert result.state["drift_decision"] is False
    assert result.state["drifted_columns"] == []
    assert result.state["reference_metadata"] == monitoring_reference.metadata.__dict__
    assert result.state["window_start"] == "2025-01-01T00:00:00Z"
    assert result.state["window_end"] == "2025-01-02T00:00:00Z"

    assert result.html_path is not None and result.html_path.is_file()
    assert result.json_path is not None and result.json_path.is_file()
    assert result.summary_path is not None and result.summary_path.is_file()
    assert result.state_path.is_file()
    report = _read_json(result.json_path)
    summary = _read_json(result.summary_path)
    state = _read_json(result.state_path)
    assert report["metrics"]
    assert summary == state["drift_summary"]
    assert summary["threshold_reached"] is False
    assert state["reports"] == {
        "html": result.html_path.relative_to(output_dir).as_posix(),
        "json": result.json_path.relative_to(output_dir).as_posix(),
        "summary": result.summary_path.relative_to(output_dir).as_posix(),
    }
    reports = cast(dict[str, object], state["reports"])
    assert all(
        isinstance(path, str) and Path(path).parts[0] == "runs"
        for path in reports.values()
    )


def test_monitoring_pipeline_detects_shifted_persisted_sample(
    tmp_path: Path,
    monitoring_reference: MonitoringReferenceResult,
) -> None:
    """Persisted, API-valid feature shifts are detected by real Evidently output."""
    database_path = tmp_path / "shifted" / "predictions.sqlite"
    output_dir = tmp_path / "shifted" / "artifacts"
    shifted = simulate_feature_drift(monitoring_reference.dataframe)
    _save_feature_events(database_path, shifted, "shifted")

    result = run_monitoring(
        repository_root=REPOSITORY_ROOT,
        database_path=database_path,
        start=WINDOW_START,
        end=WINDOW_END,
        output_dir=output_dir,
        minimum_samples=100,
    )

    assert result.status == "ready"
    assert result.state["selected_event_count"] == 3000
    assert result.state["valid_current_count"] == 3000
    assert result.state["drift_decision"] is True
    assert result.state["drifted_columns"] == [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
    ]
    summary = result.state["drift_summary"]
    assert isinstance(summary, dict)
    assert summary["drifted_columns"] == result.state["drifted_columns"]
    assert summary["threshold_reached"] is True


def test_monitoring_pipeline_filters_events_and_preserves_previous_report(
    tmp_path: Path,
    monitoring_reference: MonitoringReferenceResult,
) -> None:
    """Filtering leaves insufficient data and never publishes stale artifacts."""
    output_dir = tmp_path / "rerun" / "artifacts"
    successful_database = tmp_path / "rerun" / "successful.sqlite"
    _save_feature_events(successful_database, monitoring_reference.dataframe, "first")
    first = run_monitoring(
        REPOSITORY_ROOT,
        successful_database,
        WINDOW_START,
        WINDOW_END,
        output_dir,
        minimum_samples=100,
    )
    assert first.status == "ready"
    assert first.html_path is not None and first.html_path.is_file()
    first_report_bytes = {
        path: path.read_bytes()
        for path in (first.html_path, first.json_path, first.summary_path)
        if path is not None
    }

    database_path = tmp_path / "rerun" / "filtered.sqlite"
    valid_row = monitoring_reference.dataframe.iloc[[0]]
    _save_feature_events(database_path, valid_row, "valid")
    _save_feature_events(
        database_path,
        valid_row,
        "failed",
        status="failed",
    )
    _save_feature_events(
        database_path,
        valid_row,
        "outside",
        start=WINDOW_END + timedelta(seconds=1),
    )
    invalid_event = {
        "status": "success",
        "timestamp": "2025-01-01T00:00:10Z",
        "inputs": {**valid_row.iloc[0].to_dict(), "type": "invalid"},
    }
    save_prediction_event("invalid", invalid_event, database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO prediction_events (request_id, payload) VALUES (?, ?)",
            ("malformed", "{"),
        )

    result = run_monitoring(
        REPOSITORY_ROOT,
        database_path,
        WINDOW_START,
        WINDOW_END,
        output_dir,
        minimum_samples=100,
    )

    assert result.status == "insufficient_data"
    assert result.state["selected_event_count"] == 2
    assert result.state["valid_current_count"] == 1
    assert result.state["required_current_count"] == 100
    assert result.state["missing_current_count"] == 99
    assert result.state["skipped_invalid_records"] == 1
    assert result.state["skipped_invalid_features"] == 1
    assert result.state["drift_decision"] is None
    assert result.state["drift_summary"] is None
    assert result.state["drifted_columns"] == []
    assert result.html_path is None
    assert result.json_path is None
    assert result.summary_path is None
    assert all(path.read_bytes() == content for path, content in first_report_bytes.items())
    state = _read_json(result.state_path)
    assert state["reports"] == {"html": None, "json": None, "summary": None}
