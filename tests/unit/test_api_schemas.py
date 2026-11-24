"""Validate the distinction between training candidates and champions."""

import pytest
from pydantic import ValidationError

from predictive_maintenance.api.schemas import RegistryTrainingResponse


@pytest.mark.parametrize(
    ("champion", "previous", "promoted", "reason"),
    [
        ("2", "1", True, "higher_f1"),
        ("1", "1", False, "f1_not_improved"),
        (None, None, False, "recall_below_minimum"),
    ],
)
def test_registry_response_distinguishes_selection_outcomes(
    champion: str | None,
    previous: str | None,
    promoted: bool,
    reason: str,
) -> None:
    result = RegistryTrainingResponse(
        status="completed", training_run_id="run-2",
        model_type="gradient_boosting", model_uri="models:/m-candidate",
        metrics={"f1": 0.8}, mode="real",
        registered_model_name="predictive-maintenance-model",
        registered_version="2", champion_version=champion,
        champion_version_before=previous, promoted=promoted,
        selection_reason=reason,
    )
    payload = result.model_dump()
    assert payload["registered_version"] == "2"
    assert payload["champion_version"] == champion
    assert payload["champion_version_before"] == previous
    assert payload["promoted"] is promoted
    assert payload["selection_reason"] == reason


@pytest.mark.parametrize("invalid_version", ["0", "-1", "m-candidate", "champion"])
def test_registry_response_rejects_non_numeric_versions(invalid_version: str) -> None:
    with pytest.raises(ValidationError):
        RegistryTrainingResponse(
            status="completed", training_run_id="run-2",
            model_type="gradient_boosting", model_uri="models:/m-candidate",
            metrics={}, mode="real", registered_model_name="predictive-maintenance-model",
            registered_version=invalid_version, champion_version=None,
            champion_version_before=None, promoted=False, selection_reason="recall_below_minimum",
        )
