"""Run Registry training with verified DVC and Git provenance."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from predictive_maintenance.models.dvc_training import train_from_dvc


def main() -> None:
    """Parse explicit runtime settings and print the training result."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model using verified DVC data.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository containing data/raw.dvc.",
    )
    parser.add_argument(
        "--model-type",
        choices=["gradient_boosting", "xgboost"],
        default="gradient_boosting",
    )
    parser.add_argument(
        "--tracking-uri",
        required=True,
        help="Explicit MLflow backend URI for this training run.",
    )
    arguments = parser.parse_args()

    result = train_from_dvc(
        repository_root=arguments.repository_root.resolve(),
        model_type=arguments.model_type,
        tracking_uri=arguments.tracking_uri,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
