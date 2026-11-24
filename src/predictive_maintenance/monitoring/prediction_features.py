"""Build Evidently-compatible monitoring features from prediction events."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, cast

import pandas as pd

from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS
from predictive_maintenance.monitoring.prediction_reader import PredictionReadResult


NUMERIC_FEATURES = frozenset(FEATURE_COLUMNS[1:])
FEATURE_COLUMN_LIST = list(cast(tuple[str, ...], FEATURE_COLUMNS))
POSITIVE_FEATURES = frozenset({"air_temperature_k", "process_temperature_k"})
CATEGORICAL_VALUES = frozenset({"L", "M", "H"})


@dataclass(frozen=True)
class MonitoringFeaturesResult:
    """Monitoring feature table and the number of invalid events excluded from it."""

    dataframe: pd.DataFrame
    skipped_invalid_features: int


def _empty_features() -> pd.DataFrame:
    dataframe = pd.DataFrame(columns=pd.Index(FEATURE_COLUMN_LIST))
    return dataframe.astype({
        "type": "object",
        **{column: "float64" for column in NUMERIC_FEATURES},
    })


def _is_valid_numeric(value: object, column: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return False
    return numeric_value > 0 if column in POSITIVE_FEATURES else numeric_value >= 0


def _valid_inputs(inputs: object) -> bool:
    if not isinstance(inputs, dict) or set(inputs) != set(FEATURE_COLUMNS):
        return False
    typed_inputs = cast(dict[str, object], inputs)
    if not isinstance(typed_inputs["type"], str) or typed_inputs["type"] not in CATEGORICAL_VALUES:
        return False
    return all(_is_valid_numeric(typed_inputs[column], column) for column in NUMERIC_FEATURES)


def _features_from_event(event: dict[str, object]) -> dict[str, object] | None:
    inputs = event.get("inputs")
    if not _valid_inputs(inputs):
        return None
    inputs = cast(dict[str, object], inputs)
    return {column: inputs[column] for column in FEATURE_COLUMNS}


def build_monitoring_features(
    events: PredictionReadResult | Iterable[dict[str, object]],
) -> MonitoringFeaturesResult:
    """Build a feature-only DataFrame from events returned by the prediction reader.

    The input events are not modified. Invalid event inputs are skipped and counted
    separately from ``PredictionReadResult.skipped_invalid_records``.
    """
    source = events.events if isinstance(events, PredictionReadResult) else events
    valid_features: list[dict[str, object]] = []
    skipped_invalid_features = 0
    for event in source:
        features = _features_from_event(event)
        if features is None:
            skipped_invalid_features += 1
            continue
        valid_features.append(features)

    if not valid_features:
        dataframe = _empty_features()
    else:
        dataframe = pd.DataFrame(valid_features, columns=pd.Index(FEATURE_COLUMN_LIST))
        dataframe = dataframe.astype({column: "float64" for column in NUMERIC_FEATURES})
    return MonitoringFeaturesResult(dataframe, skipped_invalid_features)
