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


def _state_mapping(data: MonitoringDashboardData, key: str) -> dict[str, object]:
    value = data.state.get(key) if data.state is not None else None
    return value if isinstance(value, dict) else {}


def _state_text(data: MonitoringDashboardData, key: str) -> str:
    value = data.state.get(key) if data.state is not None else None
    return value if isinstance(value, str) else "Unavailable"


def _state_integer(data: MonitoringDashboardData, key: str) -> int | None:
    value = data.state.get(key) if data.state is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _render_drift_status(data: MonitoringDashboardData) -> None:
    if data.status == "insufficient_data":
        st.warning("The latest monitoring run has insufficient data.")
    elif data.status == "ready":
        decision = data.state.get("drift_decision") if data.state is not None else None
        if decision is True:
            st.error("Drift detected.")
        elif decision is False:
            st.success("No drift detected.")
        else:
            st.error("The validated drift decision is unavailable.")


def _render_runtime_data(data: MonitoringDashboardData) -> None:
    if data.state is None:
        return

    st.subheader("Monitoring run")
    st.caption(
        f"UTC window: {_state_text(data, 'window_start')} → "
        f"{_state_text(data, 'window_end')}"
    )
    st.caption(f"Generated at (UTC): {_state_text(data, 'generated_at')}")
    metrics = st.columns(4)
    metric_values = (
        ("Valid current samples", "valid_current_count"),
        ("Minimum samples", "minimum_samples"),
        ("Skipped invalid records", "skipped_invalid_records"),
        ("Skipped invalid feature rows", "skipped_invalid_features"),
    )
    for column, (label, key) in zip(metrics, metric_values):
        value = _state_integer(data, key)
        column.metric(label, value if value is not None else "Unavailable")

    if data.status == "insufficient_data":
        missing = _state_integer(data, "missing_current_count")
        st.info(
            "Missing current samples: "
            f"{missing if missing is not None else 'Unavailable'}"
        )


def _render_drifted_fields(data: MonitoringDashboardData) -> None:
    if data.status != "ready" or data.summary is None:
        return

    st.subheader("Drifted fields")
    drifted_columns = data.summary.get("drifted_columns")
    if isinstance(drifted_columns, list) and drifted_columns:
        st.write(", ".join(str(column) for column in drifted_columns))
    else:
        st.caption("No monitored fields show drift.")


def _render_reference_metadata(data: MonitoringDashboardData) -> None:
    metadata = _state_mapping(data, "reference_metadata")
    if not metadata:
        return

    with st.expander("Reference dataset version"):
        for label, key in (
            ("Source", "source_file"),
            ("DVC pointer", "dvc_pointer"),
            ("DVC hash", "raw_directory_dvc_hash"),
            ("Split policy/version", "split_policy"),
            ("Split manifest SHA-256", "split_manifest_sha256"),
            ("Reference row hash", "reference_row_hash"),
        ):
            value = metadata.get(key)
            if isinstance(value, str):
                st.caption(label)
                st.code(value, language="text")


def _render_status(data: MonitoringDashboardData) -> None:
    _render_drift_status(data)
    _render_runtime_data(data)
    _render_drifted_fields(data)
    _render_reference_metadata(data)
    if data.status == "no_run":
        st.info("No monitoring run is available yet.")
    elif data.status == "insufficient_data":
        pass
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
