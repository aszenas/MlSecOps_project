"""Extract a compact drift summary from an Evidently report."""

import json
from math import isclose, isfinite
from pathlib import Path
from typing import cast


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("ValueDrift values must be numeric.")
    try:
        numeric_value = float(value)
    except ValueError as exc:
        raise ValueError("ValueDrift values must be numeric.") from exc
    if not isfinite(numeric_value):
        raise ValueError("ValueDrift values must be finite.")
    return numeric_value


def _drifted_column(metric: object) -> str | None:
    if not isinstance(metric, dict):
        raise ValueError("Report metrics must be objects.")
    typed_metric = cast(dict[str, object], metric)
    config = typed_metric.get("config")
    if not isinstance(config, dict):
        raise ValueError("Report metric configuration must be an object.")
    typed_config = cast(dict[str, object], config)
    if typed_config.get("type") != "evidently:metric_v2:ValueDrift":
        return None

    column = typed_config.get("column")
    if not isinstance(column, str) or not column:
        raise ValueError("A ValueDrift metric has no valid column name.")
    drift_value = _finite_float(typed_metric.get("value"))
    drift_threshold = _finite_float(typed_config.get("threshold"))
    return column if drift_value >= drift_threshold else None


def _drifted_columns(metrics: object) -> list[str]:
    if not isinstance(metrics, list):
        raise ValueError("The report metrics must be a list.")
    return [
        column
        for metric in metrics
        if (column := _drifted_column(metric)) is not None
    ]


def summarize_report(report_path: Path) -> dict[str, int | float | bool | list[str]]:
    """Read and validate the aggregate drift metric."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    metrics = payload.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("The report metrics must be a list.")

    matches = [
        metric for metric in metrics
        if metric["config"]["type"]
        == "evidently:metric_v2:DriftedColumnsCount"
    ]

    if len(matches) != 1:
        raise ValueError("Expected exactly one aggregate drift metric.")

    metric = matches[0]
    columns = metric["config"]["columns"]

    if not isinstance(columns, list) or not columns:
        raise ValueError("The monitored column list is missing or empty.")

    if len(set(columns)) != len(columns):
        raise ValueError("The monitored column list contains duplicates.")

    count = float(metric["value"]["count"])
    share = float(metric["value"]["share"])
    threshold = float(metric["config"]["drift_share"])

    if not all(isfinite(value) for value in (count, share, threshold)):
        raise ValueError("Drift summary values must be finite.")

    if not count.is_integer() or not 0 <= count <= len(columns):
        raise ValueError("Invalid drifted column count.")

    if not 0 <= share <= 1 or not 0 < threshold <= 1:
        raise ValueError("Invalid drift share or decision threshold.")

    if not isclose(share, count / len(columns), abs_tol=1e-9):
        raise ValueError("Drift count and share are inconsistent.")

    return {
        "column_count": len(columns),
        "drifted_column_count": int(count),
        "drifted_column_share": share,
        "drift_share_threshold": threshold,
        "threshold_reached": share >= threshold,
        "drifted_columns": _drifted_columns(metrics),
    }


def save_summary(report_path: Path) -> Path:
    """Save the validated summary next to the Evidently report."""
    summary = summarize_report(report_path)
    summary_path = report_path.with_name("summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary_path
