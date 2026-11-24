
"""Integration tests for training with the AI4I dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.predictive_maintenance.data.preprocessing import DataProcessor
from src.predictive_maintenance.data.splitting import (
    create_initial_data_splits,
)
from src.predictive_maintenance.models.training import train_model


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "predictive_maintenance_5000.csv"
)


@pytest.fixture(scope="module")
def training_splits():
    """Load and split real predictive maintenance data."""
    if not DATASET_PATH.exists():
        pytest.skip("AI4I dataset is not available locally.")

    raw_df = pd.read_csv(DATASET_PATH)

    processor = DataProcessor()

    renamed_df = processor.rename_columns(raw_df)
    X, y = processor.split_features_target(renamed_df)

    return create_initial_data_splits(X, y)


@pytest.mark.parametrize(
    "model_type",
    ["gradient_boosting", "xgboost"],
)
def test_real_data_training(training_splits, model_type):
    """Both models should train and predict on real data."""
    processor = DataProcessor()
    preprocessor = processor.create_preprocessing_pipeline()

    pipeline = train_model(
        X_train=training_splits.X_train,
        y_train=training_splits.y_train,
        preprocessor=preprocessor,
        model_type=model_type,
    )

    predictions = pipeline.predict(
        training_splits.X_validation
    )

    probabilities = pipeline.predict_proba(
        training_splits.X_validation
    )

    assert len(predictions) == 750
    assert probabilities.shape == (750, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(np.unique(predictions)).issubset({0, 1})

    # Confirm the feature engineering and encoding are fitted.
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    transformed = fitted_preprocessor.transform(
        training_splits.X_validation
    )

    assert transformed.shape == (750, 9)
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()
