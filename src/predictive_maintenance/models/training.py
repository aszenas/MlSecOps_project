
"""Reusable model training for predictive maintenance."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from predictive_maintenance.data.imbalance import calculate_sample_weights
from xgboost import XGBClassifier


ModelType = Literal["gradient_boosting", "xgboost"]

RANDOM_STATE = 42


def build_model_pipeline(
    preprocessor: BaseEstimator,
    model_type: ModelType = "gradient_boosting",
) -> Pipeline:
    """Create an unfitted preprocessing and classification pipeline."""

    if model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        )

    elif model_type == "xgboost":
        classifier = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=1.0,
            n_jobs=1,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", classifier),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: BaseEstimator,
    model_type: ModelType = "gradient_boosting",
) -> Pipeline:
    """Fit a model using only the training partition."""

    if X_train.empty or y_train.empty:
        raise ValueError("Training data must not be empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "Features and targets must have equal lengths."
        )

    if y_train.isna().any():
        raise ValueError(
            "Training targets must not contain missing values."
        )

    labels = np.unique(y_train.to_numpy())

    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError(
            "Training targets must contain both classes: 0 and 1."
        )

    sample_weights = calculate_sample_weights(y_train)

    pipeline = build_model_pipeline(
        preprocessor=preprocessor,
        model_type=model_type,
    )

    pipeline.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weights.to_numpy(),
    )

    return pipeline
