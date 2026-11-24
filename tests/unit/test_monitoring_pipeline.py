"""Unit tests for persisted-prediction monitoring orchestration."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictive_maintenance.monitoring import monitoring_pipeline as pipeline
from predictive_maintenance.monitoring.monitoring_reference import (
    MonitoringReferenceMetadata,
    MonitoringReferenceResult,
)
from predictive_maintenance.monitoring.prediction_features import MonitoringFeaturesResult
from predictive_maintenance.monitoring.prediction_reader import (
    PredictionDatabaseError,
    PredictionReadResult,
)
from predictive_maintenance.monitoring.summary import save_summary


START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
FEATURES = pd.DataFrame({
    "type": ["L"],
    "air_temperature_k": [300.0],
    "process_temperature_k": [310.0],
    "rotational_speed_rpm": [1500.0],
    "torque_nm": [40.0],
    "tool_wear_min": [100.0],
})


def reference_result() -> MonitoringReferenceResult:
    """Build a deterministic reference result without DVC or raw data."""
    metadata = MonitoringReferenceMetadata(
        source_file="data/raw/predictive_maintenance_5000.csv",
        dvc_pointer="data/raw.dvc",
        raw_directory_dvc_hash="raw-hash",
        split_policy="initial-train-v1",
        reference_row_count=3000,
        split_manifest_sha256="manifest-hash",
        reference_row_hash="reference-hash",
    )
    return MonitoringReferenceResult(FEATURES.copy(), metadata)


def configure_pipeline(monkeypatch: pytest.MonkeyPatch, valid_count: int) -> None:
    """Stub storage, feature, and reference boundaries for one run."""
    events: list[dict[str, object]] = [
        {"request_id": str(index)} for index in range(valid_count + 2)
    ]
    read_result = PredictionReadResult(events=events, skipped_invalid_records=2)
    feature_frame = (
        pd.concat([FEATURES] * valid_count, ignore_index=True)
        if valid_count
        else FEATURES.iloc[0:0].copy()
    )
    feature_result = MonitoringFeaturesResult(
        dataframe=feature_frame,
        skipped_invalid_features=3,
    )
    monkeypatch.setattr(pipeline, "read_successful_prediction_events", lambda **kwargs: read_result)
    monkeypatch.setattr(pipeline, "build_monitoring_features", lambda result: feature_result)
    monkeypatch.setattr(pipeline, "load_monitoring_reference", lambda root: reference_result())


def report_payload() -> dict[str, object]:
    """Build an observed Evidently report payload for a fake reporter."""
    columns = [
        "type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    return {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:DriftedColumnsCount",
                    "columns": columns,
                    "drift_share": 0.5,
                },
                "value": {"count": 3, "share": 0.5},
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "type",
                    "threshold": 0.1,
                },
                "value": 0.2,
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "air_temperature_k",
                    "threshold": 0.1,
                },
                "value": 0.01,
            },
        ],
        "tests": [],
    }


def fake_reporter() -> MagicMock:
    """Create a reporter that writes all required staged artifacts."""
    reporter = MagicMock()

    def save(reference: pd.DataFrame, current: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "report.html"
        json_path = output_dir / "report.json"
        html_path.write_text("<html />", encoding="utf-8")
        json_path.write_text(json.dumps(report_payload()), encoding="utf-8")
        save_summary(json_path)
        return html_path, json_path

    reporter.save.side_effect = save
    return reporter


def read_state(output_dir: Path) -> dict[str, Any]:
    """Read the published state JSON."""
    return json.loads((output_dir / pipeline.MONITORING_STATE_FILENAME).read_text(encoding="utf-8"))


def test_insufficient_data_uses_valid_feature_count_and_skips_evidently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not report no drift when valid current samples are below threshold."""
    configure_pipeline(monkeypatch, valid_count=1)
    reporter = MagicMock()
    monkeypatch.setattr(pipeline, "DriftReporter", reporter)
    output_dir = tmp_path / "monitoring"
    output_dir.mkdir()
    previous_reports = {
        filename: output_dir / filename for filename in pipeline.REPORT_FILENAMES
    }
    for path in previous_reports.values():
        path.write_text("old", encoding="utf-8")
    previous_report_bytes = {
        path: path.read_bytes() for path in previous_reports.values()
    }

    result = pipeline.run_monitoring(
        tmp_path / "repo",
        tmp_path / "predictions.sqlite",
        START,
        END,
        output_dir,
        minimum_samples=100,
    )
    state = read_state(output_dir)

    assert result.status == "insufficient_data"
    assert state["selected_event_count"] == 3
    assert state["valid_current_count"] == 1
    assert state["skipped_invalid_records"] == 2
    assert state["skipped_invalid_features"] == 3
    assert state["drift_decision"] is None
    assert state["drift_summary"] is None
    assert state["reports"] == {"html": None, "json": None, "summary": None}
    assert all(path.read_bytes() == content for path, content in previous_report_bytes.items())
    reporter.assert_not_called()


def test_ready_state_uses_report_summary_and_publishes_all_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish a ready state only after report, JSON, and summary are complete."""
    configure_pipeline(monkeypatch, valid_count=100)
    reporter = fake_reporter()
    monkeypatch.setattr(pipeline, "DriftReporter", lambda: reporter)
    output_dir = tmp_path / "monitoring"

    result = pipeline.run_monitoring(
        tmp_path / "repo",
        tmp_path / "predictions.sqlite",
        START,
        END,
        output_dir,
        minimum_samples=100,
    )
    state = read_state(output_dir)

    assert result.status == "ready"
    assert state["status"] == "ready"
    assert state["window_start"] == "2025-01-01T00:00:00Z"
    assert state["window_end"] == "2025-01-01T01:00:00Z"
    assert state["reference_metadata"]["raw_directory_dvc_hash"] == "raw-hash"
    assert state["drift_decision"] is True
    assert state["drifted_columns"] == ["type"]
    assert state["drift_summary"]["threshold_reached"] is True
    assert all(
        (output_dir / str(path)).is_file()
        for path in state["reports"].values()
    )
    assert all(not Path(path).is_absolute() for path in state["reports"].values())
    reporter.save.assert_called_once()
    assert len(reporter.save.call_args.kwargs["current"]) == 100
    assert result.to_dict() == state


def test_successive_runs_keep_previous_reports_and_publish_new_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep published artifacts immutable when a later run succeeds."""
    configure_pipeline(monkeypatch, valid_count=100)
    reporter = fake_reporter()
    monkeypatch.setattr(pipeline, "DriftReporter", lambda: reporter)
    output_dir = tmp_path / "monitoring"

    first = pipeline.run_monitoring(
        tmp_path / "repo", tmp_path / "db", START, END, output_dir
    )
    first_state = read_state(output_dir)
    first_report_bytes = {
        path: path.read_bytes()
        for path in (first.html_path, first.json_path, first.summary_path)
        if path is not None
    }

    configure_pipeline(monkeypatch, valid_count=101)
    second = pipeline.run_monitoring(
        tmp_path / "repo", tmp_path / "db", START, END, output_dir
    )
    second_state = read_state(output_dir)

    assert first_state["reports"] != second_state["reports"]
    assert all(path.is_file() for path in first_report_bytes)
    assert all(path.read_bytes() == content for path, content in first_report_bytes.items())
    assert second.status == "ready"


def test_rejects_invalid_pipeline_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the pipeline window before reading storage."""
    reader = MagicMock()
    monkeypatch.setattr(pipeline, "read_successful_prediction_events", reader)

    with pytest.raises(ValueError, match="earlier"):
        pipeline.run_monitoring(
            tmp_path / "repo",
            tmp_path / "db",
            END,
            START,
            tmp_path / "output",
        )

    reader.assert_not_called()


def test_missing_report_artifact_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a reporter that omits the required summary artifact."""
    configure_pipeline(monkeypatch, valid_count=100)
    reporter = MagicMock()

    def incomplete_save(reference: pd.DataFrame, current: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "report.html"
        json_path = output_dir / "report.json"
        html_path.write_text("html", encoding="utf-8")
        json_path.write_text(json.dumps(report_payload()), encoding="utf-8")
        return html_path, json_path

    reporter.save.side_effect = incomplete_save
    monkeypatch.setattr(pipeline, "DriftReporter", lambda: reporter)

    with pytest.raises(RuntimeError, match="required artifacts"):
        pipeline.run_monitoring(tmp_path / "repo", tmp_path / "db", START, END, tmp_path / "output")

    assert not (tmp_path / "output" / pipeline.MONITORING_STATE_FILENAME).exists()
    runs_dir = tmp_path / "output" / "runs"
    assert not runs_dir.exists() or not list(runs_dir.iterdir())


def test_insufficient_rerun_preserves_previous_reports_without_referencing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never expose a prior successful report after a later insufficient run."""
    configure_pipeline(monkeypatch, valid_count=100)
    reporter = fake_reporter()
    monkeypatch.setattr(pipeline, "DriftReporter", lambda: reporter)
    output_dir = tmp_path / "monitoring"
    pipeline.run_monitoring(tmp_path / "repo", tmp_path / "db", START, END, output_dir)
    first_state = read_state(output_dir)
    first_report_bytes = {
        output_dir / str(path): (output_dir / str(path)).read_bytes()
        for path in first_state["reports"].values()
    }

    configure_pipeline(monkeypatch, valid_count=0)
    pipeline.run_monitoring(tmp_path / "repo", tmp_path / "db", START, END, output_dir)
    state = read_state(output_dir)

    assert state["status"] == "insufficient_data"
    assert state["drift_decision"] is None
    assert state["reports"] == {"html": None, "json": None, "summary": None}
    assert all(path.read_bytes() == content for path, content in first_report_bytes.items())
    assert reporter.save.call_count == 1


@pytest.mark.parametrize("failure", [PredictionDatabaseError("database"), ValueError("reference")])
def test_input_failures_propagate_without_success_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    """Keep database and reference failures separate from monitoring statuses."""
    if isinstance(failure, PredictionDatabaseError):
        monkeypatch.setattr(pipeline, "read_successful_prediction_events", MagicMock(side_effect=failure))
    else:
        configure_pipeline(monkeypatch, valid_count=100)
        monkeypatch.setattr(pipeline, "load_monitoring_reference", MagicMock(side_effect=failure))
    monkeypatch.setattr(pipeline, "DriftReporter", MagicMock())

    with pytest.raises(type(failure), match=str(failure)):
        pipeline.run_monitoring(tmp_path / "repo", tmp_path / "db", START, END, tmp_path / "output")

    assert not (tmp_path / "output" / pipeline.MONITORING_STATE_FILENAME).exists()


def test_report_failure_propagates_without_ready_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not publish a successful state when report generation fails."""
    configure_pipeline(monkeypatch, valid_count=100)
    reporter = MagicMock()
    reporter.save.side_effect = RuntimeError("report failed")
    monkeypatch.setattr(pipeline, "DriftReporter", lambda: reporter)

    with pytest.raises(RuntimeError, match="report failed"):
        pipeline.run_monitoring(tmp_path / "repo", tmp_path / "db", START, END, tmp_path / "output")

    assert not (tmp_path / "output" / pipeline.MONITORING_STATE_FILENAME).exists()


def test_report_failure_preserves_previous_state_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed later report generation cannot replace the current run."""
    configure_pipeline(monkeypatch, valid_count=100)
    reporter = fake_reporter()
    monkeypatch.setattr(pipeline, "DriftReporter", lambda: reporter)
    output_dir = tmp_path / "monitoring"
    first = pipeline.run_monitoring(
        tmp_path / "repo", tmp_path / "db", START, END, output_dir
    )
    state_path = output_dir / pipeline.MONITORING_STATE_FILENAME
    previous_state = state_path.read_bytes()
    previous_reports = {
        path: path.read_bytes()
        for path in (first.html_path, first.json_path, first.summary_path)
        if path is not None
    }

    configure_pipeline(monkeypatch, valid_count=101)
    reporter.save.side_effect = RuntimeError("report failed")

    with pytest.raises(RuntimeError, match="report failed"):
        pipeline.run_monitoring(
            tmp_path / "repo", tmp_path / "db", START, END, output_dir
        )

    assert state_path.read_bytes() == previous_state
    assert all(path.read_bytes() == content for path, content in previous_reports.items())
    assert len(list((output_dir / "runs").iterdir())) == 1


def test_failed_state_replace_leaves_no_partial_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove the temporary state file when atomic publication fails."""
    state_path = tmp_path / "monitoring_state.json"
    original_replace = os.replace

    def fail_state_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        if Path(destination) == state_path:
            raise OSError("state replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_state_replace)

    with pytest.raises(OSError, match="state replace failed"):
        pipeline._write_state_atomically({"status": "ready"}, state_path)

    assert not state_path.exists()
    assert not list(tmp_path.glob(".monitoring_state.json.*.tmp"))


def test_failed_state_publication_preserves_previous_state_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed state switch cannot expose the later run's report contents."""
    configure_pipeline(monkeypatch, valid_count=100)
    reporter = fake_reporter()
    monkeypatch.setattr(pipeline, "DriftReporter", lambda: reporter)
    output_dir = tmp_path / "monitoring"
    first = pipeline.run_monitoring(
        tmp_path / "repo", tmp_path / "db", START, END, output_dir
    )
    state_path = output_dir / pipeline.MONITORING_STATE_FILENAME
    previous_state = state_path.read_bytes()
    previous_reports = {
        path: path.read_bytes()
        for path in (first.html_path, first.json_path, first.summary_path)
        if path is not None
    }

    configure_pipeline(monkeypatch, valid_count=101)
    original_replace = pipeline.os.replace

    def fail_state_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        if Path(destination) == state_path:
            raise OSError("state replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_state_replace)

    with pytest.raises(OSError, match="state replace failed"):
        pipeline.run_monitoring(
            tmp_path / "repo", tmp_path / "db", START, END, output_dir
        )

    assert state_path.read_bytes() == previous_state
    assert all(path.read_bytes() == content for path, content in previous_reports.items())
    assert len(list((output_dir / "runs").iterdir())) == 1
