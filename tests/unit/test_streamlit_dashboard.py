"""Test the Streamlit dashboard shell without starting a monitoring run."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from predictive_maintenance.monitoring import dashboard_operational_data


APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_dashboard.py"


def _state(
    status: str = "ready",
    *,
    drift_decision: bool = False,
) -> dict[str, object]:
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
        "drift_decision": drift_decision if status == "ready" else None,
        "drift_summary": {
            "column_count": 6,
            "drifted_column_count": 3 if drift_decision else 0,
            "drifted_column_share": 0.5 if drift_decision else 0.0,
            "drift_share_threshold": 0.5,
            "threshold_reached": drift_decision,
            "drifted_columns": (
                ["type", "air_temperature_k", "torque_nm"]
                if drift_decision
                else []
            ),
        } if status == "ready" else None,
        "drifted_columns": (
            ["type", "air_temperature_k", "torque_nm"]
            if status == "ready" and drift_decision
            else []
        ),
        "reports": {
            "html": "report.html" if status == "ready" else None,
            "json": "report.json" if status == "ready" else None,
            "summary": "summary.json" if status == "ready" else None,
        },
    }


def _write_state(
    directory: Path,
    status: str,
    *,
    drift_decision: bool = False,
) -> None:
    directory.mkdir(exist_ok=True)
    state = _state(status, drift_decision=drift_decision)
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
    assert any(
        element.value == "A validated monitoring report is available."
        for element in app.success
    )
    assert app.get("iframe")
    assert app.download_button[0].label == "Download Evidently HTML report"
    assert app.get("metric")
    assert app.code
    assert "data/raw.dvc" in [code.value for code in app.code]


def test_dashboard_renders_detected_drift_and_drifted_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_state(tmp_path, "ready", drift_decision=True)
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert app.error[0].value == "Drift detected."
    assert any(
        "type, air_temperature_k, torque_nm" in element.value
        for element in app.markdown
    )


def test_dashboard_renders_runtime_counts_and_reference_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_state(tmp_path, "ready")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["Valid current samples"] == "120"
    assert metric_values["Minimum samples"] == "100"
    assert metric_values["Skipped invalid records"] == "0"
    assert metric_values["Skipped invalid feature rows"] == "0"
    assert "2025-01-01T00:00:00Z" in app.caption[0].value
    assert "2025-01-02T01:00:00Z" in app.caption[1].value
    assert "a" * 32 + ".dir" in [code.value for code in app.code]


def test_dashboard_renders_insufficient_data_without_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_state(tmp_path, "insufficient_data")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert app.warning[0].value == "The latest monitoring run has insufficient data."
    assert any("Missing current samples: 100" in element.value for element in app.info)
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


def test_refresh_does_not_keep_previous_report_or_status(
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

    _write_state(tmp_path, "ready", drift_decision=True)
    app.run()
    assert app.get("iframe")

    (tmp_path / "report.html").unlink()
    app.run()
    assert any("Monitoring data unavailable:" in item.value for item in app.error)
    assert not app.get("iframe")

    _write_state(tmp_path, "ready")
    app.run()
    assert app.get("iframe")


def test_dashboard_renders_operational_sources_independently(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_champion_dashboard_data",
        lambda: dashboard_operational_data.ChampionDashboardData(
            status="ready",
            model_name="predictive-maintenance-model",
            version="9",
            queried_at="2026-01-01T00:00:00+00:00",
        ),
    )
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_retraining_dashboard_data",
        lambda: dashboard_operational_data.RetrainingDashboardData(
            status="unavailable",
            error="backend offline",
        ),
    )

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert any("Champion version: 9" in item.value for item in app.success)
    assert any("backend offline" in item.value for item in app.error)


def test_dashboard_renders_missing_champion_and_recorded_retraining(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_champion_dashboard_data",
        lambda: dashboard_operational_data.ChampionDashboardData(
            status="not_found", error="No champion model is available."
        ),
    )
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_retraining_dashboard_data",
        lambda: dashboard_operational_data.RetrainingDashboardData(
            status="ready",
            record=dashboard_operational_data.RetrainingRecord(
                run_id="run-1",
                completed_at="2026-01-01T00:00:00+00:00",
                result="rejected",
                challenger_version="4",
                promotion_decision="not_promoted",
            ),
        ),
    )

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not app.exception
    assert any("No champion model is available." in item.value for item in app.warning)
    assert any("Recorded result: rejected" in item.value for item in app.info)
    assert any("Promotion decision: not_promoted" in item.value for item in app.caption)


def test_dashboard_keeps_operational_data_visible_when_monitoring_is_invalid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "monitoring_state.json").write_text("{", encoding="utf-8")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_champion_dashboard_data",
        lambda: dashboard_operational_data.ChampionDashboardData(
            status="ready", model_name="model", version="3"
        ),
    )
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_retraining_dashboard_data",
        lambda: dashboard_operational_data.RetrainingDashboardData(
            status="no_data"
        ),
    )

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert any("Monitoring data unavailable:" in item.value for item in app.error)
    assert any("Champion version: 3" in item.value for item in app.success)
    assert any("No recorded retraining result" in item.value for item in app.info)


def test_dashboard_keeps_ready_report_when_mlflow_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_state(tmp_path, "ready")
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_champion_dashboard_data",
        lambda: dashboard_operational_data.ChampionDashboardData(
            status="unavailable", error="registry offline"
        ),
    )
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_retraining_dashboard_data",
        lambda: dashboard_operational_data.RetrainingDashboardData(
            status="unavailable", error="experiment offline"
        ),
    )

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert app.get("iframe")
    assert any("registry offline" in item.value for item in app.error)
    assert any("experiment offline" in item.value for item in app.error)


def test_dashboard_refresh_clears_old_operational_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MONITORING_REPORT_DIR", str(tmp_path))
    current = {"value": True}

    def champion_loader():
        if current["value"]:
            return dashboard_operational_data.ChampionDashboardData(
                status="ready", model_name="model", version="11"
            )
        return dashboard_operational_data.ChampionDashboardData(
            status="unavailable", error="refresh failed"
        )

    monkeypatch.setattr(
        dashboard_operational_data, "load_champion_dashboard_data", champion_loader
    )
    monkeypatch.setattr(
        dashboard_operational_data,
        "load_retraining_dashboard_data",
        lambda: dashboard_operational_data.RetrainingDashboardData(
            status="not_configured", error="not configured"
        ),
    )

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert any("Champion version: 11" in item.value for item in app.success)

    current["value"] = False
    app.run()

    assert not any("Champion version: 11" in item.value for item in app.success)
    assert any("refresh failed" in item.value for item in app.error)
