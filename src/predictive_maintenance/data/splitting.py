
"""Reproducible stratified data splitting for predictive maintenance."""

from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, replace


RANDOM_STATE = 42

TRAIN_SIZE = 3000
VALIDATION_SIZE = 750
FUTURE_SIZE = 750
FINAL_TEST_SIZE = 500
V2_TRAINING_UDIS = (16, 18, 24, 27, 29, 40, 45, 51, 69, 77)

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

    if bool(y.isna().any()) or not bool(y.isin([0, 1]).all()):
        raise ValueError("Target must contain only 0 and 1 without missing values.")

    if y.nunique() != 2:
        raise ValueError("Both target classes must be present.")

    # Final test set
    X_remaining, X_test, y_remaining, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X,
            y,
            test_size=FINAL_TEST_SIZE,
            stratify=y,
            random_state=random_state,
        ),
    )

    # Data for future observations, monitoring and drift simulation
    X_remaining, X_future, y_remaining, y_future = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X_remaining,
            y_remaining,
            test_size=FUTURE_SIZE,
            stratify=y_remaining,
            random_state=random_state,
        ),
    )

    # Validation + initial training partitions
    X_train, X_validation, y_train, y_validation = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X_remaining,
            y_remaining,
            test_size=VALIDATION_SIZE,
            stratify=y_remaining,
            random_state=random_state,
        ),
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


def create_retraining_data_splits(
    initial_splits: DataSplits,
    source_udis: pd.Series,
    training_udis: tuple[int, ...] = V2_TRAINING_UDIS,
) -> DataSplits:
    """Move selected labeled future records into the training partition."""
    if not training_udis or len(set(training_udis)) != len(training_udis):
        raise ValueError("Retraining UDIs must be non-empty and unique.")

    if not source_udis.index.is_unique or not source_udis.is_unique or bool(source_udis.isna().any()):
        raise ValueError("Source UDIs must be non-missing and unique, with a unique index.")

    future_udis = source_udis.reindex(initial_splits.X_future.index)
    if bool(future_udis.isna().any()):
        raise ValueError("Every future row must have a source UDI.")

    rows_by_udi = dict(zip(future_udis.tolist(), future_udis.index, strict=True))
    invalid_udis = sorted(set(training_udis).difference(rows_by_udi))
    if invalid_udis:
        raise ValueError(f"Retraining UDIs must belong to the future split: {invalid_udis}.")

    moved_rows = [rows_by_udi[udi] for udi in training_udis]
    return replace(
        initial_splits,
        X_train=pd.concat([initial_splits.X_train, initial_splits.X_future.loc[moved_rows]]),
        y_train=pd.concat([initial_splits.y_train, initial_splits.y_future.loc[moved_rows]]),
        X_future=initial_splits.X_future.drop(index=moved_rows),
        y_future=initial_splits.y_future.drop(index=moved_rows),
    )
