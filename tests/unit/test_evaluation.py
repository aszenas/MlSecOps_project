
"""Unit tests for binary classification evaluation."""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.predictive_maintenance.models.evaluation import evaluate_model


class DummyModel:
    """Return predetermined probability predictions."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return self.probabilities


@pytest.fixture
def evaluation_data():
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    probabilities = [
        [0.9, 0.1],
        [0.6, 0.4],
        [0.3, 0.7],
        [0.2, 0.8],
    ]

    return X, y, probabilities


def make_model(probabilities):
    """Adapt a dummy predictor for the evaluation interface."""
    return cast(Pipeline, DummyModel(probabilities))


def test_evaluate_perfect_predictions(evaluation_data):
    X, y, probabilities = evaluation_data

    metrics = evaluate_model(
        make_model(probabilities),
        X,
        y,
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }

    assert all(value == 1.0 for value in metrics.values())


def test_evaluate_custom_threshold(evaluation_data):
    X, y, probabilities = evaluation_data

    metrics = evaluate_model(
        make_model(probabilities),
        X,
        y,
        threshold=0.3,
    )

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0


def test_evaluate_rejects_empty_data():
    X = pd.DataFrame({"feature": []})
    y = pd.Series([], dtype=int)

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_model(make_model([]), X, y)


def test_evaluate_rejects_mismatched_lengths(evaluation_data):
    X, y, probabilities = evaluation_data

    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_model(
            make_model(probabilities),
            X.iloc[:3],
            y,
        )


def test_evaluate_rejects_single_class(evaluation_data):
    X, _, probabilities = evaluation_data
    y = pd.Series([0, 0, 0, 0])

    with pytest.raises(ValueError, match="both binary classes"):
        evaluate_model(make_model(probabilities), X, y)


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 0.0, 1.0, 1.1, float("nan")],
)
def test_evaluate_rejects_invalid_threshold(
    evaluation_data,
    threshold,
):
    X, y, probabilities = evaluation_data

    with pytest.raises(ValueError, match="Threshold"):
        evaluate_model(
            make_model(probabilities),
            X,
            y,
            threshold=threshold,
        )


def test_evaluate_rejects_invalid_probabilities(evaluation_data):
    X, y, _ = evaluation_data

    with pytest.raises(ValueError, match="two-class"):
        evaluate_model(
            make_model([[0.5]]),
            X,
            y,
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_evaluate_rejects_non_finite_probabilities(
    evaluation_data,
    invalid_probability,
):
    """Reject NaN and infinite prediction probabilities."""
    X, y, _ = evaluation_data

    probabilities = [
        [0.9, 0.1],
        [0.6, invalid_probability],
        [0.3, 0.7],
        [0.2, 0.8],
    ]

    with pytest.raises(
        ValueError,
        match="Predicted probabilities must be finite",
    ):
        evaluate_model(
            make_model(probabilities),
            X,
            y,
        )
