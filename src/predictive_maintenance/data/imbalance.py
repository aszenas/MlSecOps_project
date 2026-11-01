"""Class imbalance analysis and training sample weights."""

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight


def validate_binary_target(y: pd.Series) -> None:
    """Validate a non-empty binary classification target."""
    if y.empty:
        raise ValueError("Target cannot be empty.")

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if y.nunique() != 2:
        raise ValueError("Both target classes must be present.")


def create_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Calculate class counts and proportions for a target series."""
    validate_binary_target(y)

    counts = y.value_counts().sort_index()
    ratios = y.value_counts(normalize=True).sort_index() * 100

    return pd.DataFrame(
        {
            "count": counts,
            "ratio_percent": ratios.round(2),
        }
    )


def calculate_sample_weights(y_train: pd.Series) -> pd.Series:
    """Calculate balanced sample weights from training labels only."""
    validate_binary_target(y_train)

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train.to_numpy(),
    )

    return pd.Series(
        weights,
        index=y_train.index,
        name="sample_weight",
        dtype=float,
    )
