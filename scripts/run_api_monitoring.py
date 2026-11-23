"""Generate monitoring reports from persisted API prediction events."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from predictive_maintenance.monitoring.monitoring_pipeline import run_monitoring
from predictive_maintenance.monitoring.sample_status import (
    DEFAULT_MINIMUM_CURRENT_SAMPLES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_timestamp(value: str) -> datetime:
    """Require an ISO 8601 timestamp with an explicit timezone."""
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use an ISO 8601 timestamp, such as 2026-11-23T00:00:00Z."
        ) from exc

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("The timestamp must include a timezone.")

    return timestamp.astimezone(timezone.utc)


def positive_integer(value: str) -> int:
    """Require a positive sample threshold."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("The value must be greater than zero.")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Define monitoring inputs without starting a monitoring run."""
    repository_root = Path(
        os.environ.get("TRAINING_REPOSITORY_ROOT", str(PROJECT_ROOT))
    )
    database_path = Path(
        os.environ.get(
            "PREDICTION_LOG_DB",
            str(PROJECT_ROOT / "artifacts" / "predictions.db"),
        )
    )
    reports_root = Path(
        os.environ.get("REPORTS_DIR", str(PROJECT_ROOT / "reports"))
    )
    output_dir = Path(
        os.environ.get("MONITORING_REPORT_DIR", str(reports_root / "monitoring"))
    )

    parser = argparse.ArgumentParser(
        description="Monitor successful API predictions in a selected time window.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root,
        help="Repository containing the verified raw DVC dataset.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=database_path,
        help="SQLite prediction event database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Directory for monitoring state and report artifacts.",
    )
    parser.add_argument(
        "--since",
        type=parse_timestamp,
        required=True,
        help="Inclusive window start with timezone.",
    )
    parser.add_argument(
        "--until",
        type=parse_timestamp,
        required=True,
        help="Exclusive window end with timezone.",
    )
    parser.add_argument(
        "--minimum-samples",
        type=positive_integer,
        default=DEFAULT_MINIMUM_CURRENT_SAMPLES,
        help="Minimum valid prediction rows required for drift evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run monitoring and print its published state."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.since >= arguments.until:
        parser.error("--since must be earlier than --until.")

    try:
        result = run_monitoring(
            repository_root=arguments.repository_root.resolve(),
            database_path=arguments.database.resolve(),
            start=arguments.since,
            end=arguments.until,
            output_dir=arguments.output_dir.resolve(),
            minimum_samples=arguments.minimum_samples,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Monitoring failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
