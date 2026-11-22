"""Minimal Streamlit entrypoint for persisted monitoring data."""

import os
from pathlib import Path

import streamlit as st

from predictive_maintenance.monitoring.dashboard_data import (
    MonitoringDashboardData,
    load_monitoring_dashboard_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY_ENV = "MONITORING_REPORT_DIR"
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "monitoring"


def configured_report_directory() -> Path:
    """Return the configured report directory, resolved from the project root."""
    configured = os.environ.get(REPORT_DIRECTORY_ENV)
    path = Path(configured) if configured else DEFAULT_REPORT_DIRECTORY
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _render_status(data: MonitoringDashboardData) -> None:
    if data.status == "no_run":
        st.info("No monitoring run is available yet.")
    elif data.status == "insufficient_data":
        st.warning("The latest monitoring run has insufficient data.")
    elif data.status == "ready":
        if data.html is None:
            st.error("The validated monitoring report content is unavailable.")
            return
        st.success("A validated monitoring report is available.")
        st.download_button(
            "Download Evidently HTML report",
            data=data.html,
            file_name="report.html",
            mime="text/html",
            on_click="ignore",
        )
        st.iframe(
            data.html,
            width="stretch",
            height=800,
            alt="Evidently monitoring report",
        )
    else:
        message = data.error or "The monitoring data is temporarily unavailable."
        st.error(f"Monitoring data unavailable: {message}")


def main() -> None:
    """Render the monitoring dashboard shell without starting monitoring."""
    st.set_page_config(
        page_title="Predictive Maintenance Monitoring",
        page_icon="📈",
    )
    st.title("Predictive Maintenance Monitoring")
    st.write("View the latest persisted monitoring state and report status.")

    st.button("Refresh monitoring data", help="Read the monitoring files again.")
    try:
        data = load_monitoring_dashboard_data(configured_report_directory())
    except OSError as exc:
        data = MonitoringDashboardData(
            status="unavailable",
            error=f"The monitoring data could not be read: {exc}",
        )
    _render_status(data)


if __name__ == "__main__":
    main()
