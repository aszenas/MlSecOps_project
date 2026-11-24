"""Run monitoring and prediction export periodically."""

import argparse
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import FrameType

from predictive_maintenance.api.services import (
    get_tracking_uri,
    get_training_repository_root,
)
from predictive_maintenance.monitoring.automation import run_monitoring_cycle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_integer(value: str) -> int:
    """Require a positive interval or sample count."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("The value must be greater than zero.")
    return number


def run_once(
    window_seconds: int,
    minimum_samples: int,
) -> dict[str, object]:
    """Process the latest monitoring window using server settings."""
    end = datetime.now(timezone.utc)
    reports_root = Path(
        os.environ.get("REPORTS_DIR", str(PROJECT_ROOT / "reports"))
    )
    output_dir = Path(
        os.environ.get("MONITORING_REPORT_DIR", str(reports_root / "monitoring"))
    )
    database_path = Path(
        os.environ.get(
            "PREDICTION_LOG_DB",
            str(PROJECT_ROOT / "artifacts" / "predictions.db"),
        )
    )

    enabled = os.environ.get(
        "AUTOMATIC_RETRAINING_ENABLED", "false",
    ).strip().lower()
    if enabled not in {"true", "false"}:
        raise ValueError("AUTOMATIC_RETRAINING_ENABLED must be true or false.")

    return run_monitoring_cycle(
        repository_root=get_training_repository_root(),
        database_path=database_path.resolve(),
        start=end - timedelta(seconds=window_seconds),
        end=end,
        output_dir=output_dir.resolve(),
        tracking_uri=get_tracking_uri(),
        minimum_samples=minimum_samples,
        retraining_enabled=enabled == "true",
    )


def run_worker(
    interval_seconds: int,
    window_seconds: int,
    minimum_samples: int,
    stop_event: Event,
    once: bool = False,
) -> int:
    """Run sequential cycles until stopped, retrying after handled failures."""
    while not stop_event.is_set():
        try:
            result = run_once(window_seconds, minimum_samples)
            print(json.dumps(result, allow_nan=False), flush=True)
            exit_code = 0 if result["status"] == "completed" else 1
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Monitoring cycle failed: {exc}", file=sys.stderr, flush=True)
            exit_code = 1

        if once:
            return exit_code

        stop_event.wait(interval_seconds)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Read configuration and handle shutdown requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--interval-seconds",
        type=positive_integer,
        default=os.environ.get("MONITORING_INTERVAL_SECONDS", "300"),
    )
    parser.add_argument(
        "--window-seconds",
        type=positive_integer,
        default=os.environ.get("MONITORING_WINDOW_SECONDS", "3600"),
    )
    parser.add_argument(
        "--minimum-samples",
        type=positive_integer,
        default=os.environ.get("MONITORING_MINIMUM_SAMPLES", "100"),
    )
    arguments = parser.parse_args(argv)
    stop_event = Event()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    return run_worker(
        interval_seconds=arguments.interval_seconds,
        window_seconds=arguments.window_seconds,
        minimum_samples=arguments.minimum_samples,
        stop_event=stop_event,
        once=arguments.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
