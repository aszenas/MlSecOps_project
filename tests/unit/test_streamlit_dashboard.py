"""Test the Streamlit dashboard shell without starting a monitoring run."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_dashboard.py"


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
