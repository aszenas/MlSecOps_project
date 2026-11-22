"""Test the Streamlit dashboard shell without starting a monitoring run."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_dashboard.py"


def _state(status: str = "ready") -> dict[str, object]:
    selected_event_count = 2 if status == "insufficient_data" else 120
    valid_current_count = 0 if status == "insufficient_data" else 120
    return {
        "schema_version": 1,
        "status": status,
        "window_start": "2025-01-01T00:00:00Z",
        "window_end": "2025-01-02T00:00:00Z",
        "generated_at": "2025-01-02T01:00:00Z",
        "minimum_samples": 100,
        "selected_event_count": selected_event_count,
        "valid_current_count": valid_current_count,
        "required_current_count": 100,
        "missing_current_count": max(100 - valid_current_count, 0),
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
        "drift_decision": False if status == "ready" else None,
        "drift_summary": {
            "column_count": 6,
            "drifted_column_count": 0,
            "drifted_column_share": 0.0,
            "drift_share_threshold": 0.5,
            "threshold_reached": False,
            "drifted_columns": [],
        } if status == "ready" else None,
        "drifted_columns": [],
        "reports": {
            "html": "report.html" if status == "ready" else None,
            "json": "report.json" if status == "ready" else None,
            "summary": "summary.json" if status == "ready" else None,
        },
    }


def _write_state(directory: Path, status: str) -> None:
    directory.mkdir(exist_ok=True)
    state = _state(status)
    if status == "ready":
        (directory / "report.html").write_text(
            "<html><body><script>window.reportReady = true;</script></body></html>",
            encoding="utf-8",
        )
        (directory / "report.json").write_text(
            json.dumps({"metrics": [{"config": {"type": "report"}}]}),
            encoding="utf-8",
        )
        (directory / "summary.json").write_text(
            json.dumps(state["drift_summary"]),
            encoding="utf-8",
        )
    (directory / "monitoring_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )


def test_dashboard_renders_title_and_no_run_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "Predictive Maintenance Monitoring"
    assert app.info[0].value == "No monitoring run is available yet."
    assert app.button[0].label == "Refresh monitoring data"


def test_dashboard_renders_ready_report_and_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_state(tmp_path, "ready")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert app.success[0].value == "A validated monitoring report is available."
    assert app.get("iframe")
    assert app.download_button[0].label == "Download Evidently HTML report"


def test_dashboard_renders_insufficient_data_without_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_state(tmp_path, "insufficient_data")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert app.warning[0].value == "The latest monitoring run has insufficient data."
    assert not app.get("iframe")


def test_dashboard_renders_unavailable_without_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "monitoring_state.json").write_text("{", encoding="utf-8")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert app.error[0].value.startswith("Monitoring data unavailable:")
    assert not app.get("iframe")


def test_refresh_does_not_keep_previous_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_state(tmp_path, "ready")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert app.get("iframe")

    _write_state(tmp_path, "insufficient_data")
    app.run()

    assert not app.exception
    assert app.warning[0].value == "The latest monitoring run has insufficient data."
    assert not app.get("iframe")
