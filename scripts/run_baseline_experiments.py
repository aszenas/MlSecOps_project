
"""Run baseline training experiments on the AI4I dataset."""

from pathlib import Path

import pandas as pd

from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.splitting import create_initial_data_splits
from predictive_maintenance.models.mlflow_tracking import (
    run_training_experiment,
)


def main() -> None:
    """Load the dataset and log both baseline models."""
    dataset_path = Path("data/raw/predictive_maintenance_5000.csv")

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run this script from the project root."
        )

    raw_data = pd.read_csv(dataset_path)
    processor = DataProcessor()

    normalized_data = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized_data)
    splits = create_initial_data_splits(features, target)

    for model_type in ("gradient_boosting", "xgboost"):
        result = run_training_experiment(
            splits=splits,
            model_type=model_type,
        )

        print(f"\nModel: {result['model_type']}")
        print(f"Run ID: {result['run_id']}")
        print(f"Model URI: {result['model_uri']}")

        metrics = result["metrics"]
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
