
"""Unit tests for automatic champion selection."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from predictive_maintenance.models import model_selection


@pytest.mark.parametrize(
    ("challenger_f1", "challenger_recall", "champion_f1", "expected"),
    [
        (0.80, 0.75, 0.70, True),
        (0.60, 0.75, 0.70, False),
        (0.70, 0.75, 0.70, False),
        (0.90, 0.69, 0.70, False),
        (0.80, 0.70, 0.70, True),
    ],
)
def test_should_promote_with_existing_champion(
    challenger_f1: float,
    challenger_recall: float,
    champion_f1: float,
    expected: bool,
) -> None:
    """Enforce the F1 and minimum recall requirements."""
    challenger = {
        "f1": challenger_f1,
        "recall": challenger_recall,
    }
    champion = {"f1": champion_f1, "recall": 0.75}

    assert model_selection.should_promote(
        challenger,
        champion,
    ) is expected


@pytest.mark.parametrize(
    ("recall", "expected"),
    [
        (0.70, True),
        (0.90, True),
        (0.69, False),
    ],
)
def test_first_champion_requires_minimum_recall(
    recall: float,
    expected: bool,
) -> None:
    """Reject the first candidate if recall is insufficient."""
    metrics = {"f1": 0.80, "recall": recall}

    assert model_selection.should_promote(
        metrics,
        None,
    ) is expected


@pytest.mark.parametrize(
    "minimum_recall",
    [0.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_should_promote_rejects_invalid_minimum(
    minimum_recall: float,
) -> None:
    """Reject invalid minimum recall thresholds."""
    with pytest.raises(ValueError, match="Minimum recall"):
        model_selection.should_promote(
            {"f1": 0.8, "recall": 0.8},
            None,
            minimum_recall,
        )


@pytest.mark.parametrize(
    "metrics",
    [
        {"f1": float("nan"), "recall": 0.8},
        {"f1": 0.8, "recall": float("inf")},
        {"f1": 1.1, "recall": 0.8},
        {"recall": 0.8},
    ],
)
def test_should_promote_rejects_bad_metrics(
    metrics: dict[str, float],
) -> None:
    """Reject missing or invalid comparison metrics."""
    with pytest.raises(ValueError, match="Invalid F1-score"):
        model_selection.should_promote(metrics, None)


@pytest.mark.parametrize(
    ("champion_version", "challenger_f1", "challenger_recall",
     "expected_promotion", "expected_reason"),
    [
        (None, 0.8, 0.75, True, "first_champion"),
        (None, 0.8, 0.60, False, "recall_below_minimum"),
        ("1", 0.8, 0.75, True, "higher_f1"),
        ("1", 0.6, 0.75, False, "f1_not_improved"),
        ("1", 0.9, 0.60, False, "recall_below_minimum"),
        ("2", 0.8, 0.75, False, "already_champion"),
    ],
)
def test_evaluate_and_select_champion(
    monkeypatch: pytest.MonkeyPatch,
    champion_version: str | None,
    challenger_f1: float,
    challenger_recall: float,
    expected_promotion: bool,
    expected_reason: str,
) -> None:
    """Evaluate candidates and update the alias only when eligible."""
    client = MagicMock()
    client.get_registered_model.return_value.aliases = (
        {} if champion_version is None
        else {"champion": champion_version}
    )

    monkeypatch.setattr(
        model_selection,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    loader = MagicMock(spec=model_selection._load_registered_version)
    loader.return_value = Pipeline(steps=[])
    monkeypatch.setattr(
        model_selection,
        "_load_registered_version",
        loader,
    )

    evaluator = MagicMock()
    evaluator.side_effect = [
        {"f1": challenger_f1, "recall": challenger_recall},
        {"f1": 0.70, "recall": 0.75},
    ]
    monkeypatch.setattr(model_selection, "evaluate_model", evaluator)

    promoter = MagicMock()
    monkeypatch.setattr(
        model_selection,
        "set_champion_version",
        promoter,
    )

    X = pd.DataFrame({"feature": [1, 2]})
    y = pd.Series([0, 1])

    result = model_selection.evaluate_and_select_champion(
        "2",
        X,
        y,
    )

    assert result.promoted is expected_promotion
    assert result.reason == expected_reason

    if expected_promotion:
        assert result.selected_version == "2"
        promoter.assert_called_once_with(
            "2",
            model_selection.REGISTERED_MODEL_NAME,
        )
    else:
        assert result.selected_version == champion_version
        promoter.assert_not_called()

    expected_evaluations = 1 if champion_version is None else 2
    assert evaluator.call_count == expected_evaluations

    for call in evaluator.call_args_list:
        assert call.args[1] is X
        assert call.args[2] is y
        assert call.kwargs["threshold"] == 0.5


@pytest.mark.parametrize(
    "version",
    ["", "0", "-1", "abc"],
)
def test_selection_rejects_invalid_version(version: str) -> None:
    """Reject invalid challenger version identifiers."""
    with pytest.raises(ValueError, match="Challenger version"):
        model_selection.evaluate_and_select_champion(
            version,
            pd.DataFrame(),
            pd.Series(dtype=int),
        )


def test_selection_rejects_empty_model_name() -> None:
    """Reject blank registered model names."""
    with pytest.raises(ValueError, match="model name"):
        model_selection.evaluate_and_select_champion(
            "1",
            pd.DataFrame(),
            pd.Series(dtype=int),
            registered_model_name=" ",
        )


def test_selection_propagates_loading_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not promote when model loading fails."""
    client = MagicMock()
    client.get_registered_model.return_value.aliases = {}

    monkeypatch.setattr(
        model_selection,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    def fail_to_load(name: str, version: str) -> Pipeline:
        raise RuntimeError("Cannot load candidate")

    monkeypatch.setattr(
        model_selection,
        "_load_registered_version",
        fail_to_load,
    )

    with pytest.raises(RuntimeError, match="Cannot load candidate"):
        model_selection.evaluate_and_select_champion(
            "1",
            pd.DataFrame(),
            pd.Series(dtype=int),
        )


def test_load_registered_version_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load a specific registered model version as a Pipeline."""
    expected_model = Pipeline(steps=[])
    loader = MagicMock(return_value=expected_model)

    monkeypatch.setattr(
        model_selection.mlflow_sklearn,
        "load_model",
        loader,
    )

    result = model_selection._load_registered_version(
        "predictive-maintenance-model",
        "2",
    )

    assert result is expected_model
    loader.assert_called_once_with(
        "models:/predictive-maintenance-model/2"
    )


def test_load_registered_version_rejects_invalid_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a registered artifact that is not an sklearn Pipeline."""
    loader = MagicMock(return_value={"invalid": "model"})

    monkeypatch.setattr(
        model_selection.mlflow_sklearn,
        "load_model",
        loader,
    )

    with pytest.raises(
        ValueError,
        match="The loaded model must be an sklearn Pipeline",
    ):
        model_selection._load_registered_version(
            "predictive-maintenance-model",
            "2",
        )

    loader.assert_called_once_with(
        "models:/predictive-maintenance-model/2"
    )


@pytest.mark.parametrize(
    "minimum_recall",
    [0.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_selection_rejects_invalid_minimum_recall(
    minimum_recall: float,
) -> None:
    """Validate the recall guardrail before accessing the Registry."""
    X = pd.DataFrame({"feature": [1, 2]})
    y = pd.Series([0, 1])

    with pytest.raises(
        ValueError,
        match="Minimum recall must be in",
    ):
        model_selection.evaluate_and_select_champion(
            challenger_version="2",
            X_validation=X,
            y_validation=y,
            minimum_recall=minimum_recall,
        )
