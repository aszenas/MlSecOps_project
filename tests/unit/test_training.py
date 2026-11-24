
"""Unit tests for predictive maintenance model training."""

from typing import cast
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.predictive_maintenance.models.training import (
    build_model_pipeline,
    train_model,
)


@pytest.fixture
def training_data():
    """Generate synthetic imbalanced binary classification data."""
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "feature_1": rng.normal(size=100),
        "feature_2": rng.normal(size=100),
    })

    y = pd.Series([0] * 90 + [1] * 10)

    return X, y


@pytest.mark.parametrize(
    "model_type,expected_class",
    [
        ("gradient_boosting", GradientBoostingClassifier),
        ("xgboost", XGBClassifier),
    ],
)
def test_build_pipeline(model_type, expected_class):
    pipeline = build_model_pipeline(
        StandardScaler(),
        model_type=model_type,
    )

    assert list(pipeline.named_steps) == [
        "preprocessor",
        "model",
    ]
    assert isinstance(
        pipeline.named_steps["model"],
        expected_class,
    )


@pytest.mark.parametrize(
    "model_type",
    ["gradient_boosting", "xgboost"],
)
def test_train_model_returns_fitted_pipeline(
    training_data,
    model_type,
):
    X, y = training_data

    pipeline = train_model(
        X,
        y,
        StandardScaler(),
        model_type=model_type,
    )

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(np.unique(predictions)).issubset({0, 1})


@pytest.mark.parametrize(
    "model_type",
    ["gradient_boosting", "xgboost"],
)
def test_training_is_reproducible(training_data, model_type):
    X, y = training_data

    first = train_model(
        X, y, StandardScaler(), model_type=model_type
    )
    second = train_model(
        X, y, StandardScaler(), model_type=model_type
    )

    np.testing.assert_array_equal(
        first.predict(X),
        second.predict(X),
    )


def test_rejects_empty_training_data():
    X = pd.DataFrame({"feature": []})
    y = pd.Series([], dtype=int)

    with pytest.raises(ValueError, match="must not be empty"):
        train_model(X, y, StandardScaler())


def test_rejects_mismatched_lengths():
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="equal lengths"):
        train_model(X, y, StandardScaler())


def test_rejects_missing_target():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, np.nan])

    with pytest.raises(ValueError, match="missing values"):
        train_model(X, y, StandardScaler())


def test_rejects_single_class():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0])

    with pytest.raises(ValueError, match="both classes"):
        train_model(X, y, StandardScaler())


def test_rejects_invalid_model_type():
    from src.predictive_maintenance.models.training import ModelType

    with pytest.raises(ValueError, match="Unsupported model"):
        build_model_pipeline(
            StandardScaler(),
            model_type=cast(ModelType, "invalid"),
        )
