"""Create deterministic persisted-prediction monitoring demonstrations."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import cast

import pandas as pd

from predictive_maintenance.monitoring.monitoring_pipeline import (
    MonitoringRunResult,
    run_monitoring,
)
from predictive_maintenance.monitoring.monitoring_reference import (
    load_monitoring_reference,
)
from predictive_maintenance.monitoring.prediction_store import (
    save_prediction_event,
)
from predictive_maintenance.monitoring.simulation import simulate_feature_drift


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2025, 1, 2, tzinfo=timezone.utc)
MINIMUM_SAMPLES = 100


def _python_value(value: object) -> object:
    item = getattr(value, "item", None)
    return cast(object, item()) if callable(item) else value


def _save_feature_events(
    database_path: Path,
    features: pd.DataFrame,
    request_prefix: str,
) -> int:
    count = 0
    for count, (_, row) in enumerate(features.iterrows(), start=1):
        inputs = {
            column: _python_value(row[column])
            for column in features.columns
        }
        save_prediction_event(
            request_id=f"{request_prefix}-{count - 1}",
            event={
                "status": "success",
                "timestamp": (
                    WINDOW_START + timedelta(seconds=count)
                ).isoformat().replace("+00:00", "Z"),
                "inputs": inputs,
            },
            database_path=database_path,
        )
    return count


def _run_scenario(
    repository_root: Path,
    output_root: Path,
    name: str,
    features: pd.DataFrame,
    request_prefix: str,
) -> MonitoringRunResult:
    scenario_root = output_root / name
    database_path = scenario_root / "predictions.sqlite"
    report_directory = scenario_root / "reports"
    _save_feature_events(database_path, features, request_prefix)
    return run_monitoring(
        repository_root=repository_root,
        database_path=database_path,
        start=WINDOW_START,
        end=WINDOW_END,
        output_dir=report_directory,
        minimum_samples=MINIMUM_SAMPLES,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run normal, shifted and insufficient persisted monitoring demos."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository containing the verified DVC dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory for demo databases and monitoring reports.",
    )
    return parser.parse_args()


def main() -> None:
    """Persist synthetic API-shaped events and run all three scenarios."""
    arguments = _arguments()
    repository_root = arguments.repository_root.resolve()
    output_root = arguments.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    reference = load_monitoring_reference(repository_root)
    normal = _run_scenario(
        repository_root,
        output_root,
        "normal",
        reference.dataframe,
        "normal",
    )
    shifted = _run_scenario(
        repository_root,
        output_root,
        "shifted",
        simulate_feature_drift(reference.dataframe),
        "shifted",
    )
    insufficient = _run_scenario(
        repository_root,
        output_root,
        "insufficient_data",
        reference.dataframe.iloc[:1],
        "insufficient",
    )

    print(json.dumps({
        "normal": normal.to_dict(),
        "shifted": shifted.to_dict(),
        "insufficient_data": insufficient.to_dict(),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
