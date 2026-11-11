
"""Automatic champion selection for predictive maintenance."""

from dataclasses import dataclass
from math import isfinite

import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from predictive_maintenance.models.evaluation import evaluate_model
from predictive_maintenance.models.model_registry import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    set_champion_version,
)


MINIMUM_RECALL = 0.70


@dataclass(frozen=True)
class SelectionResult:
    """Store champion selection results and the decision reason."""

    promoted: bool
    reason: str
    selected_version: str | None
    challenger_version: str
    champion_version_before: str | None
    challenger_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None


def should_promote(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    minimum_recall: float = MINIMUM_RECALL,
) -> bool:
    """Select only an eligible challenger with a higher F1-score."""
    if not isfinite(minimum_recall) or not 0.0 < minimum_recall <= 1.0:
        raise ValueError("Minimum recall must be in (0, 1].")

    for metrics in (challenger_metrics, champion_metrics):
        if metrics is None:
            continue

        if (
            "f1" not in metrics
            or "recall" not in metrics
            or not all(
                isfinite(metrics[key])
                and 0.0 <= metrics[key] <= 1.0
                for key in ("f1", "recall")
            )
        ):
            raise ValueError("Invalid F1-score or recall metrics.")

    if challenger_metrics["recall"] < minimum_recall:
        return False

    if champion_metrics is None:
        return True

    return challenger_metrics["f1"] > champion_metrics["f1"]


def _load_registered_version(
    name: str,
    version: str,
) -> Pipeline:
    """Load a specific version, not a mutable alias."""
    model_uri = f"models:/{name}/{version}"
    model = mlflow_sklearn.load_model(model_uri)

    if not isinstance(model, Pipeline):
        raise ValueError("The loaded model must be an sklearn Pipeline.")

    return model


def evaluate_and_select_champion(
    challenger_version: str,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    minimum_recall: float = MINIMUM_RECALL,
    threshold: float = 0.5,
    apply_promotion: bool = True,
) -> SelectionResult:
    """Compare registered versions and promote only a better challenger."""
    if not registered_model_name.strip():
        raise ValueError("Registered model name must not be empty.")

    if (
        not challenger_version.isascii()
        or not challenger_version.isdecimal()
        or int(challenger_version) < 1
    ):
        raise ValueError("Challenger version must be a positive integer.")

    if not isfinite(minimum_recall) or not 0.0 < minimum_recall <= 1.0:
        raise ValueError("Minimum recall must be in (0, 1].")

    client = MlflowClient()
    registered_model = client.get_registered_model(
        registered_model_name
    )
    current_version = registered_model.aliases.get(CHAMPION_ALIAS)

    challenger = _load_registered_version(
        registered_model_name,
        challenger_version,
    )
    challenger_metrics = evaluate_model(
        challenger,
        X_validation,
        y_validation,
        threshold=threshold,
    )

    champion_metrics = None
    if current_version is not None:
        champion = _load_registered_version(
            registered_model_name,
            str(current_version),
        )
        champion_metrics = evaluate_model(
            champion,
            X_validation,
            y_validation,
            threshold=threshold,
        )

    promoted = should_promote(
        challenger_metrics,
        champion_metrics,
        minimum_recall,
    )

    if current_version is not None and (
        str(current_version) == challenger_version
    ):
        promoted = False
        reason = "already_champion"
    elif challenger_metrics["recall"] < minimum_recall:
        reason = "recall_below_minimum"
    elif promoted:
        reason = "first_champion" if current_version is None else "higher_f1"
    else:
        reason = "f1_not_improved"

    if promoted and apply_promotion:
        set_champion_version(
            challenger_version,
            registered_model_name,
        )

    selected_version = (
        challenger_version
        if promoted
        else str(current_version) if current_version is not None else None
    )

    return SelectionResult(
        promoted=promoted,
        reason=reason,
        selected_version=selected_version,
        challenger_version=challenger_version,
        champion_version_before=(
            str(current_version) if current_version is not None else None
        ),
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
    )
