"""Run monitoring from persisted prediction events and publish its state."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from predictive_maintenance.monitoring.drift_report import DriftReporter
from predictive_maintenance.monitoring.monitoring_reference import (
    MonitoringReferenceMetadata,
    load_monitoring_reference,
)
from predictive_maintenance.monitoring.prediction_features import build_monitoring_features
from predictive_maintenance.monitoring.prediction_reader import (
    read_successful_prediction_events,
)
from predictive_maintenance.monitoring.sample_status import (
    DEFAULT_MINIMUM_CURRENT_SAMPLES,
    MonitoringSampleResult,
    assess_monitoring_sample_count,
)
from predictive_maintenance.monitoring.summary import summarize_report


MONITORING_STATE_SCHEMA_VERSION = 1
MONITORING_STATE_FILENAME = "monitoring_state.json"
REPORT_FILENAMES = ("report.html", "report.json", "summary.json")


@dataclass(frozen=True)
class MonitoringRunResult:
    """Published monitoring outcome and its JSON-serializable state."""

    status: str
    state_path: Path
    html_path: Path | None
    json_path: Path | None
    summary_path: Path | None
    state: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return the monitoring state without converting filesystem paths."""
        return dict(self.state)


def _normalize_timestamp(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"The {label} timestamp must include a timezone.")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _metadata_dict(metadata: MonitoringReferenceMetadata) -> dict[str, object]:
    return cast(dict[str, object], asdict(metadata))


def _report_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return tuple(output_dir / filename for filename in REPORT_FILENAMES)  # type: ignore[return-value]


def _remove_previous_reports(output_dir: Path) -> None:
    for path in _report_paths(output_dir):
        path.unlink(missing_ok=True)


def _write_state_atomically(state: dict[str, object], state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_path.parent,
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(state, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, state_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _base_state(
    start: datetime,
    end: datetime,
    generated_at: datetime,
    minimum_samples: int,
    selected_event_count: int,
    valid_current_count: int,
    sample_result: MonitoringSampleResult,
    skipped_invalid_records: int,
    skipped_invalid_features: int,
    reference_metadata: MonitoringReferenceMetadata,
) -> dict[str, object]:
    return {
        "schema_version": MONITORING_STATE_SCHEMA_VERSION,
        "status": sample_result.status,
        "window_start": _timestamp_text(start),
        "window_end": _timestamp_text(end),
        "generated_at": _timestamp_text(generated_at),
        "minimum_samples": minimum_samples,
        "selected_event_count": selected_event_count,
        "valid_current_count": valid_current_count,
        "required_current_count": sample_result.required_count,
        "missing_current_count": sample_result.missing_count,
        "skipped_invalid_records": skipped_invalid_records,
        "skipped_invalid_features": skipped_invalid_features,
        "reference_metadata": _metadata_dict(reference_metadata),
        "drift_decision": None,
        "drift_summary": None,
        "drifted_columns": [],
        "reports": {"html": None, "json": None, "summary": None},
    }


def _result_from_state(
    state: dict[str, object],
    state_path: Path,
) -> MonitoringRunResult:
    reports = cast(dict[str, object], state["reports"])
    return MonitoringRunResult(
        status=cast(str, state["status"]),
        state_path=state_path,
        html_path=Path(value) if isinstance((value := reports.get("html")), str) else None,
        json_path=Path(value) if isinstance((value := reports.get("json")), str) else None,
        summary_path=Path(value) if isinstance((value := reports.get("summary")), str) else None,
        state=state,
    )


def run_monitoring(
    repository_root: Path,
    database_path: Path,
    start: datetime,
    end: datetime,
    output_dir: Path,
    minimum_samples: int = DEFAULT_MINIMUM_CURRENT_SAMPLES,
) -> MonitoringRunResult:
    """Generate and publish monitoring state from persisted prediction events."""
    normalized_start = _normalize_timestamp(start, "start")
    normalized_end = _normalize_timestamp(end, "end")
    if normalized_start >= normalized_end:
        raise ValueError("The start timestamp must be earlier than the end timestamp.")

    read_result = read_successful_prediction_events(
        database_path=database_path,
        start=normalized_start,
        end=normalized_end,
    )
    feature_result = build_monitoring_features(read_result)
    sample_result = assess_monitoring_sample_count(
        len(feature_result.dataframe),
        minimum_samples=minimum_samples,
    )
    reference_result = load_monitoring_reference(repository_root)
    output_dir = Path(output_dir).resolve()
    state_path = output_dir / MONITORING_STATE_FILENAME
    generated_at = datetime.now(timezone.utc)
    state = _base_state(
        start=normalized_start,
        end=normalized_end,
        generated_at=generated_at,
        minimum_samples=sample_result.required_count,
        selected_event_count=len(read_result.events),
        valid_current_count=len(feature_result.dataframe),
        sample_result=sample_result,
        skipped_invalid_records=read_result.skipped_invalid_records,
        skipped_invalid_features=feature_result.skipped_invalid_features,
        reference_metadata=reference_result.metadata,
    )

    if sample_result.status == "insufficient_data":
        _remove_previous_reports(output_dir)
        _write_state_atomically(state, state_path)
        return _result_from_state(state, state_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".monitoring-report-", dir=output_dir))
    try:
        staged_html, staged_json = DriftReporter().save(
            reference=reference_result.dataframe,
            current=feature_result.dataframe,
            output_dir=staging_dir,
        )
        staged_summary = staged_json.with_name("summary.json")
        summary = summarize_report(staged_json)
        if not staged_html.is_file() or not staged_json.is_file() or not staged_summary.is_file():
            raise RuntimeError("The monitoring report did not produce all required artifacts.")

        final_html, final_json, final_summary = _report_paths(output_dir)
        for source, destination in (
            (staged_html, final_html),
            (staged_json, final_json),
            (staged_summary, final_summary),
        ):
            os.replace(source, destination)

        state["drift_decision"] = summary["threshold_reached"]
        state["drift_summary"] = summary
        state["drifted_columns"] = summary["drifted_columns"]
        state["reports"] = {
            "html": str(final_html),
            "json": str(final_json),
            "summary": str(final_summary),
        }
        _write_state_atomically(state, state_path)
        return _result_from_state(state, state_path)
    finally:
        for path in staging_dir.iterdir():
            path.unlink(missing_ok=True)
        staging_dir.rmdir()
