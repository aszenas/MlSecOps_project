
"""Model evaluation utilities for binary classification."""

from typing import Any, cast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a fitted binary classifier on validation data."""

    if X_validation.empty or y_validation.empty:
        raise ValueError("Validation data must not be empty.")

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "Features and targets must have equal lengths."
        )

    if (
        y_validation.isna().any()
        or not y_validation.isin([0, 1]).all()
        or y_validation.nunique() != 2
    ):
        raise ValueError(
            "Validation targets must contain both binary classes."
        )

    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    probabilities = np.asarray(
        model.predict_proba(X_validation)
    )

    if probabilities.shape != (len(y_validation), 2):
        raise ValueError("Expected two-class probability predictions.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities must be finite.")

    positive_probabilities = probabilities[:, 1]

    predictions = (
        positive_probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_validation, predictions)
        ),
        "precision": float(
            precision_score(
                y_validation,
                predictions,
                zero_division=cast(Any, 0),
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                predictions,
                zero_division=cast(Any, 0),
            )
        ),
        "f1": float(
            f1_score(
                y_validation,
                predictions,
                zero_division=cast(Any, 0),
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_validation,
                positive_probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_validation,
                positive_probabilities,
            )
        ),
    }

    return metrics
