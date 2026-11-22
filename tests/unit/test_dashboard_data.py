"""Test validation and loading of persisted dashboard monitoring data."""

import json
from pathlib import Path
from typing import cast

from predictive_maintenance.monitoring.dashboard_data import (
    load_monitoring_dashboard_data,
)


def _state(status: str = "ready") -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "window_start": "2025-01-01T00:00:00Z",
        "window_end": "2025-01-02T00:00:00Z",
        "generated_at": "2025-01-02T01:00:00Z",
        "minimum_samples": 100,
        "selected_event_count": 120,
        "valid_current_count": 120,
        "required_current_count": 100,
        "missing_current_count": 0,
        "skipped_invalid_records": 0,
        "skipped_invalid_features": 0,
        "reference_metadata": {
            "source_file": "data/raw/predictive_maintenance_5000.csv",
            "dvc_pointer": "data/raw.dvc",
            "raw_directory_dvc_hash": "a" * 32 + ".dir",
            "split_policy": "initial",
            "reference_row_count": 3000,
            "split_manifest_sha256": "b" * 64,
            "reference_row_hash": "c" * 64,
        },
        "drift_decision": False,
        "drift_summary": {
            "column_count": 6,
            "drifted_column_count": 0,
            "drifted_column_share": 0.0,
            "drift_share_threshold": 0.5,
            "threshold_reached": False,
            "drifted_columns": [],
        },
        "drifted_columns": [],
        "reports": {
            "html": "report.html",
            "json": "report.json",
            "summary": "summary.json",
        },
    }


def _write_ready_reports(
    directory: Path,
    report_subdirectory: Path = Path(),
    *,
    marker: str = "report",
    drift_decision: bool = False,
) -> None:
    state = _state()
    report_directory = directory / report_subdirectory
    report_directory.mkdir(parents=True, exist_ok=True)
    if drift_decision:
        state["drift_decision"] = True
        state["drift_summary"] = {
            "column_count": 6,
            "drifted_column_count": 3,
            "drifted_column_share": 0.5,
            "drift_share_threshold": 0.5,
            "threshold_reached": True,
            "drifted_columns": ["type", "air_temperature_k", "torque_nm"],
        }
        state["drifted_columns"] = state["drift_summary"]["drifted_columns"]
    state["reports"] = {
        name: (report_subdirectory / filename).as_posix()
        for name, filename in (
            ("html", "report.html"),
            ("json", "report.json"),
            ("summary", "summary.json"),
        )
    }
    (report_directory / "report.html").write_text(
        f"<html>{marker}</html>", encoding="utf-8"
    )
    report = {"metrics": [{"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}}]}
    (report_directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (report_directory / "summary.json").write_text(
        json.dumps(state["drift_summary"]),
        encoding="utf-8",
    )
    (directory / "monitoring_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )


def test_missing_state_means_no_run(tmp_path: Path) -> None:
    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "no_run"
    assert result.state is None
    assert result.error is None


def test_valid_ready_state_loads_all_reports_and_is_json_serializable(
    tmp_path: Path,
) -> None:
    _write_ready_reports(tmp_path)

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "ready"
    assert result.state is not None
    assert result.report == {
        "metrics": [{"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}}]
    }
    assert result.summary == _state()["drift_summary"]
    assert result.html == "<html>report</html>"
    json.dumps(result.to_dict(), allow_nan=False)


def test_loads_only_the_current_report_from_versioned_run_directory(
    tmp_path: Path,
) -> None:
    _write_ready_reports(tmp_path, Path("runs/previous"), marker="previous")
    _write_ready_reports(
        tmp_path,
        Path("runs/current"),
        marker="current",
        drift_decision=True,
    )

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "ready"
    assert result.html == "<html>current</html>"
    assert result.state is not None
    assert result.state["drift_decision"] is True
    reports = cast(dict[str, object], result.state["reports"])
    assert reports["html"] == "runs/current/report.html"
    assert (tmp_path / "runs/previous/report.html").read_text(encoding="utf-8") == (
        "<html>previous</html>"
    )


def test_failed_new_publication_keeps_loader_on_previous_run(
    tmp_path: Path,
) -> None:
    _write_ready_reports(tmp_path, Path("runs/original"), marker="original")
    previous_state = (tmp_path / "monitoring_state.json").read_bytes()
    _write_ready_reports(tmp_path, Path("runs/unpublished"), marker="unpublished")
    (tmp_path / "monitoring_state.json").write_bytes(previous_state)

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "ready"
    assert result.html == "<html>original</html>"
    assert result.state is not None
    assert result.state["drift_decision"] is False
    reports = cast(dict[str, object], result.state["reports"])
    assert reports["html"] == "runs/original/report.html"


def test_insufficient_state_does_not_show_retained_previous_report(
    tmp_path: Path,
) -> None:
    _write_ready_reports(tmp_path, Path("runs/previous"), marker="previous")
    state = _state("insufficient_data")
    state.update({
        "selected_event_count": 2,
        "valid_current_count": 0,
        "missing_current_count": 100,
        "drift_decision": None,
        "drift_summary": None,
        "reports": {"html": None, "json": None, "summary": None},
    })
    (tmp_path / "monitoring_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "insufficient_data"
    assert result.html is None
    assert result.report is None
    assert (tmp_path / "runs/previous/report.html").is_file()


def test_valid_insufficient_data_has_no_reports(tmp_path: Path) -> None:
    state = _state("insufficient_data")
    state.update({
        "selected_event_count": 2,
        "valid_current_count": 0,
        "missing_current_count": 100,
        "drift_decision": None,
        "drift_summary": None,
        "reports": {"html": None, "json": None, "summary": None},
    })
    tmp_path.joinpath("monitoring_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "insufficient_data"
    assert result.state == state
    assert result.report is None
    assert result.html is None


def test_missing_report_does_not_become_ready(tmp_path: Path) -> None:
    _write_ready_reports(tmp_path)
    (tmp_path / "report.html").unlink()

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "unavailable"
    assert result.state is None
    assert result.error


def test_corrupt_state_json_is_unavailable(tmp_path: Path) -> None:
    (tmp_path / "monitoring_state.json").write_text("{", encoding="utf-8")

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "unavailable"
    assert result.error


def test_state_and_summary_mismatch_is_unavailable(tmp_path: Path) -> None:
    _write_ready_reports(tmp_path)
    state = _state()
    cast_summary = state["drift_summary"]
    assert isinstance(cast_summary, dict)
    cast_summary["threshold_reached"] = True
    (tmp_path / "monitoring_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "unavailable"
    assert result.error


def test_report_path_escape_is_unavailable(tmp_path: Path) -> None:
    _write_ready_reports(tmp_path)
    state = _state()
    reports = state["reports"]
    assert isinstance(reports, dict)
    reports["html"] = "../outside.html"
    (tmp_path / "monitoring_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    result = load_monitoring_dashboard_data(tmp_path)

    assert result.status == "unavailable"
    assert result.error
