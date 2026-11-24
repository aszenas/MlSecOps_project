"""Load validated monitoring data for a future dashboard consumer."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from predictive_maintenance.monitoring.monitoring_pipeline import (
    MONITORING_STATE_FILENAME,
    MONITORING_STATE_SCHEMA_VERSION,
)


DashboardDataStatus = Literal["no_run", "insufficient_data", "ready", "unavailable"]


@dataclass(frozen=True)
class MonitoringDashboardData:
    """Validated dashboard data or an explicit non-ready outcome."""

    status: DashboardDataStatus
    state: dict[str, object] | None = None
    report: dict[str, object] | None = None
    summary: dict[str, object] | None = None
    html: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the loader result."""
        return {
            "status": self.status,
            "state": self.state,
            "report": self.report,
            "summary": self.summary,
            "html": self.html,
            "error": self.error,
        }


def _unavailable(message: str) -> MonitoringDashboardData:
    return MonitoringDashboardData(status="unavailable", error=message)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return cast(dict[str, object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if value < 1 if positive else value < 0:
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {requirement}.")
    return value


def _timestamp(value: object, label: str) -> datetime:
    text = _string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO 8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone.")
    return parsed


def _reject_nonstandard_json(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}.")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path.name}.") from exc
    return _mapping(payload, path.name)


def _resolve_report_path(report_dir: Path, value: object, label: str) -> Path:
    text = _string(value, label)
    if "://" in text:
        raise ValueError(f"{label} must refer to a local file.")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = report_dir / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(report_dir)
    except ValueError as exc:
        raise ValueError(f"{label} points outside the report directory.") from exc
    return resolved


def _validate_reference_metadata(value: object) -> None:
    metadata = _mapping(value, "reference_metadata")
    for field in (
        "source_file",
        "dvc_pointer",
        "raw_directory_dvc_hash",
        "split_policy",
        "split_manifest_sha256",
        "reference_row_hash",
    ):
        _string(metadata.get(field), f"reference_metadata.{field}")
    _integer(metadata.get("reference_row_count"), "reference_metadata.reference_row_count", positive=True)


def _validate_summary(value: object) -> dict[str, object]:
    summary = _mapping(value, "summary")
    column_count = _integer(summary.get("column_count"), "summary.column_count", positive=True)
    drifted_count = _integer(summary.get("drifted_column_count"), "summary.drifted_column_count")
    if drifted_count > column_count:
        raise ValueError("summary.drifted_column_count exceeds column_count.")

    share = summary.get("drifted_column_share")
    threshold = summary.get("drift_share_threshold")
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
        raise ValueError("summary.drifted_column_share is invalid.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError("summary.drift_share_threshold is invalid.")

    reached = summary.get("threshold_reached")
    if not isinstance(reached, bool) or reached != (share >= threshold):
        raise ValueError("summary.threshold_reached is inconsistent.")

    columns = summary.get("drifted_columns")
    if not isinstance(columns, list) or not all(isinstance(column, str) and column for column in columns):
        raise ValueError("summary.drifted_columns is invalid.")
    if len(columns) != drifted_count or len(set(columns)) != len(columns):
        raise ValueError("summary drifted column count is inconsistent.")
    return summary


def _validate_state(state: dict[str, object]) -> tuple[str, dict[str, object]]:
    required_fields = {
        "schema_version",
        "status",
        "window_start",
        "window_end",
        "generated_at",
        "minimum_samples",
        "selected_event_count",
        "valid_current_count",
        "required_current_count",
        "missing_current_count",
        "skipped_invalid_records",
        "skipped_invalid_features",
        "reference_metadata",
        "drift_decision",
        "drift_summary",
        "drifted_columns",
        "reports",
    }
    if not required_fields.issubset(state):
        raise ValueError("The monitoring state is missing required fields.")
    schema_version = state.get("schema_version")
    if schema_version != MONITORING_STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported monitoring state schema version.")

    start = _timestamp(state.get("window_start"), "window_start")
    end = _timestamp(state.get("window_end"), "window_end")
    _timestamp(state.get("generated_at"), "generated_at")
    if start >= end:
        raise ValueError("The monitoring window is invalid.")

    minimum = _integer(state.get("minimum_samples"), "minimum_samples", positive=True)
    selected = _integer(state.get("selected_event_count"), "selected_event_count")
    valid = _integer(state.get("valid_current_count"), "valid_current_count")
    required = _integer(state.get("required_current_count"), "required_current_count", positive=True)
    missing = _integer(state.get("missing_current_count"), "missing_current_count")
    _integer(state.get("skipped_invalid_records"), "skipped_invalid_records")
    _integer(state.get("skipped_invalid_features"), "skipped_invalid_features")
    if minimum != required or valid > selected or missing != max(required - valid, 0):
        raise ValueError("Monitoring sample counts are inconsistent.")
    _validate_reference_metadata(state.get("reference_metadata"))

    status = state.get("status")
    if status not in ("ready", "insufficient_data"):
        raise ValueError("Unsupported monitoring state status.")
    reports = _mapping(state.get("reports"), "reports")
    if set(reports) != {"html", "json", "summary"}:
        raise ValueError("The reports object has an invalid shape.")
    return cast(str, status), reports


def _load_insufficient_state(
    state: dict[str, object],
    reports: dict[str, object],
) -> MonitoringDashboardData:
    if any(value is not None for value in reports.values()):
        raise ValueError("insufficient_data must not reference reports.")
    if state.get("drift_decision") is not None or state.get("drift_summary") is not None:
        raise ValueError("insufficient_data must not contain a drift result.")
    if state.get("drifted_columns") != []:
        raise ValueError("insufficient_data must not contain drifted columns.")
    return MonitoringDashboardData(status="insufficient_data", state=state)


def _load_ready_state(
    root: Path,
    state: dict[str, object],
    reports: dict[str, object],
) -> MonitoringDashboardData:
    if not isinstance(state.get("drift_decision"), bool):
        raise ValueError("ready must contain a boolean drift_decision.")
    summary = _validate_summary(state.get("drift_summary"))
    if state["drift_decision"] != summary["threshold_reached"]:
        raise ValueError("State and summary drift decisions are inconsistent.")
    if state.get("drifted_columns") != summary["drifted_columns"]:
        raise ValueError("State and summary drifted columns are inconsistent.")

    html_path = _resolve_report_path(root, reports["html"], "reports.html")
    json_path = _resolve_report_path(root, reports["json"], "reports.json")
    summary_path = _resolve_report_path(root, reports["summary"], "reports.summary")
    report = _read_json(json_path)
    saved_summary = _read_json(summary_path)
    if saved_summary != summary:
        raise ValueError("State and summary report are inconsistent.")
    if not isinstance(report.get("metrics"), list) or not report["metrics"]:
        raise ValueError("The Evidently report has no metrics list.")
    html = html_path.read_text(encoding="utf-8")
    if not html:
        raise ValueError("The HTML report is empty.")
    return MonitoringDashboardData(
        status="ready",
        state=state,
        report=report,
        summary=summary,
        html=html,
    )


def load_monitoring_dashboard_data(report_dir: Path) -> MonitoringDashboardData:
    """Load and validate the persisted monitoring state and its reports.

    ``report_dir`` is the only filesystem root used for report resolution. The
    function never starts monitoring or modifies any loaded file.
    """
    root = Path(report_dir).resolve()
    state_path = root / MONITORING_STATE_FILENAME
    if not state_path.exists():
        return MonitoringDashboardData(status="no_run")
    if not state_path.is_file():
        return _unavailable("The monitoring state path is not a file.")

    try:
        state = _read_json(state_path)
        status, reports = _validate_state(state)
        if status == "insufficient_data":
            return _load_insufficient_state(state, reports)
        return _load_ready_state(root, state, reports)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return _unavailable(str(exc))
