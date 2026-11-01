
"""Reproducible stratified data splitting for predictive maintenance."""

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

TRAIN_SIZE = 3000
VALIDATION_SIZE = 750
FUTURE_SIZE = 750
FINAL_TEST_SIZE = 500

TOTAL_SIZE = (
    TRAIN_SIZE + VALIDATION_SIZE + FUTURE_SIZE + FINAL_TEST_SIZE
)


@dataclass(frozen=True)
class DataSplits:
    """Store features and targets for the four dataset partitions."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_future: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_future: pd.Series
    y_test: pd.Series


def create_initial_data_splits(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Create four disjoint, reproducible, stratified partitions."""
    if len(X) != TOTAL_SIZE or len(y) != TOTAL_SIZE:
        raise ValueError(
            f"Expected {TOTAL_SIZE} records in X and y. "
            f"Received X={len(X)}, y={len(y)}."
        )

    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indexes.")

    if not X.index.is_unique:
        raise ValueError("Dataset indexes must be unique.")

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1 without missing values.")

    if y.nunique() != 2:
        raise ValueError("Both target classes must be present.")

    # Final test set
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=FINAL_TEST_SIZE,
        stratify=y,
        random_state=random_state,
    )

    # Data for future observations, monitoring and drift simulation
    X_remaining, X_future, y_remaining, y_future = train_test_split(
        X_remaining,
        y_remaining,
        test_size=FUTURE_SIZE,
        stratify=y_remaining,
        random_state=random_state,
    )

    # Validation + initial training partitions
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_remaining,
        y_remaining,
        test_size=VALIDATION_SIZE,
        stratify=y_remaining,
        random_state=random_state,
    )

    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_future=X_future,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_future=y_future,
        y_test=y_test,
    )
