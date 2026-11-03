
"""Verify an MLflow model against reproducible training."""

import argparse
from pathlib import Path
from typing import cast

import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.splitting import create_initial_data_splits
from predictive_maintenance.models.evaluation import evaluate_model
from predictive_maintenance.models.training import ModelType, train_model


DATASET_PATH = Path("data/raw/predictive_maintenance_5000.csv")


def verify_model(model_uri: str, model_type: ModelType) -> None:
    """Compare an MLflow model with a reproducibly trained model."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Recreate the original data preparation and split.
    processor = DataProcessor()
    raw_data = pd.read_csv(DATASET_PATH)
    normalized_data = processor.rename_columns(raw_data)
    X, y = processor.split_features_target(normalized_data)
    splits = create_initial_data_splits(X, y)

    # Train a fresh reference model with the same configuration.
    reference_model = train_model(
        X_train=splits.X_train,
        y_train=splits.y_train,
        preprocessor=processor.create_preprocessing_pipeline(),
        model_type=model_type,
    )

    # Load the previously logged model from MLflow.
    loaded_model = mlflow_sklearn.load_model(model_uri)

    if not isinstance(loaded_model, Pipeline):
        raise TypeError("Loaded MLflow model must be a sklearn Pipeline.")

    X_validation = splits.X_validation
    y_validation = splits.y_validation

    # Compare class predictions.
    expected_predictions = reference_model.predict(X_validation)
    loaded_predictions = loaded_model.predict(X_validation)

    np.testing.assert_array_equal(
        expected_predictions,
        loaded_predictions,
    )

    # Compare probabilities, including floating-point tolerance.
    expected_probabilities = reference_model.predict_proba(
        X_validation
    )
    loaded_probabilities = loaded_model.predict_proba(
        X_validation
    )

    np.testing.assert_allclose(
        expected_probabilities,
        loaded_probabilities,
        rtol=1e-10,
        atol=1e-10,
    )

    # Compare evaluation metrics.
    reference_metrics = evaluate_model(
        reference_model, X_validation, y_validation
    )
    loaded_metrics = evaluate_model(
        loaded_model, X_validation, y_validation
    )

    for metric_name in reference_metrics:
        np.testing.assert_allclose(
            reference_metrics[metric_name],
            loaded_metrics[metric_name],
            rtol=1e-10,
            atol=1e-10,
        )

    print(f"\nModel: {model_type}")
    print(f"MLflow URI: {model_uri}")
    print(f"Validation samples: {len(X_validation)}")
    print("Class predictions: MATCH")
    print("Predicted probabilities: MATCH")
    print("Evaluation metrics: MATCH")
    print("RESULT: MODEL VERIFICATION PASSED")


def main() -> None:
    """Parse command-line arguments and run verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument(
        "--model-type",
        choices=["gradient_boosting", "xgboost"],
        required=True,
    )
    args = parser.parse_args()

    verify_model(
        model_uri=args.model_uri,
        model_type=cast(ModelType, args.model_type),
    )


if __name__ == "__main__":
    main()
