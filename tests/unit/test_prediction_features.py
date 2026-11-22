"""Unit tests for monitoring feature construction."""

import copy
from typing import Any

import pytest

from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS
from predictive_maintenance.monitoring.prediction_features import (
    build_monitoring_features,
)
from predictive_maintenance.monitoring.prediction_reader import PredictionReadResult


def valid_event() -> dict[str, object]:
    """Build a valid event containing the six API input features."""
    return {
        "status": "success",
        "inputs": {
            "type": "M",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 100.0,
        },
    }


def test_builds_ordered_feature_table_with_expected_dtypes() -> None:
    """Return exactly the six model inputs in their canonical order."""
    result = build_monitoring_features([valid_event()])

    assert list(result.dataframe.columns) == list(FEATURE_COLUMNS)
    assert result.dataframe.dtypes.to_dict() == {
        "type": "object",
        "air_temperature_k": "float64",
        "process_temperature_k": "float64",
        "rotational_speed_rpm": "float64",
        "torque_nm": "float64",
        "tool_wear_min": "float64",
    }
    assert result.skipped_invalid_features == 0


def test_empty_input_has_expected_schema() -> None:
    """Keep an empty feature table usable for the sample-count decision."""
    result = build_monitoring_features([])

    assert result.dataframe.empty
    assert list(result.dataframe.columns) == list(FEATURE_COLUMNS)
    assert result.dataframe.dtypes["type"] == "object"
    assert all(
        result.dataframe.dtypes[column] == "float64"
        for column in FEATURE_COLUMNS[1:]
    )
    assert result.skipped_invalid_features == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda inputs: inputs.pop("torque_nm"),
        lambda inputs: inputs.update(type="X"),
        lambda inputs: inputs.update(rotational_speed_rpm=True),
        lambda inputs: inputs.update(air_temperature_k=float("nan")),
        lambda inputs: inputs.update(process_temperature_k=float("inf")),
        lambda inputs: inputs.update(torque_nm=-1.0),
        lambda inputs: inputs.update(tool_wear_min=-1.0),
        lambda inputs: inputs.update(extra=1),
    ],
)
def test_skips_invalid_feature_records(mutate: Any) -> None:
    """Reject schema, category, boolean, finite-value, and range violations."""
    invalid = valid_event()
    inputs = invalid["inputs"]
    assert isinstance(inputs, dict)
    mutate(inputs)

    result = build_monitoring_features([invalid])

    assert result.dataframe.empty
    assert result.skipped_invalid_features == 1


def test_record_and_feature_skips_are_separate() -> None:
    """Preserve reader-level skips separately from feature-level skips."""
    valid = valid_event()
    invalid = valid_event()
    assert isinstance(invalid["inputs"], dict)
    invalid["inputs"]["type"] = "X"
    read_result = PredictionReadResult(
        events=[valid, invalid],
        skipped_invalid_records=4,
    )

    result = build_monitoring_features(read_result)

    assert len(result.dataframe) == 1
    assert result.skipped_invalid_features == 1
    assert read_result.skipped_invalid_records == 4


def test_does_not_modify_input_events() -> None:
    """Build features without mutating nested event dictionaries."""
    events = [valid_event()]
    original = copy.deepcopy(events)

    build_monitoring_features(events)

    assert events == original
