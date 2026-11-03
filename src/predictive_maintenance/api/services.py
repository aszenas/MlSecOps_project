"""Connect API operations to the existing machine learning pipeline."""

from pathlib import Path

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.splitting import create_initial_data_splits
from predictive_maintenance.models.mlflow_tracking import run_training_experiment
from predictive_maintenance.models.training import ModelType


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_5000.csv"


def train_pipeline(
    model_type: ModelType = "gradient_boosting",
) -> dict[str, object]:
    """Train and log a model using the initial dataset and fixed splits."""
    raw_data = load_raw_data(DATASET_PATH)

    processor = DataProcessor()
    normalized_data = processor.rename_columns(raw_data)
    features, target = processor.split_features_target(normalized_data)

    splits = create_initial_data_splits(features, target)

    return run_training_experiment(
        splits=splits,
        model_type=model_type,
    )
