
"""Unit tests for Evidently drift report orchestration."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictive_maintenance.data.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)
from predictive_maintenance.monitoring import drift_report


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide reference and current machine data."""
    reference = pd.DataFrame(
        {
            "type": ["L", "M"],
            "air_temperature_k": [300.0, 301.0],
            "process_temperature_k": [310.0, 311.0],
            "rotational_speed_rpm": [1500, 1600],
            "torque_nm": [40.0, 45.0],
            "tool_wear_min": [100, 120],
        }
    )

    current = reference.copy(deep=True)
    current["air_temperature_k"] += 10.0

    return reference, current


@pytest.fixture
def mocked_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    """Replace external report and preprocessing dependencies."""
    mocks: dict[str, MagicMock] = {}

    for name in (
        "DataProcessor",
        "DataDefinition",
        "Dataset",
        "DataSummaryPreset",
        "DataDriftPreset",
        "Report",
        "save_summary",
    ):
        mock = MagicMock()
        monkeypatch.setattr(drift_report, name, mock)
        mocks[name] = mock

    processor = mocks["DataProcessor"].return_value
    processor.select_features.side_effect = (
        lambda data: data.copy(deep=True)
    )

    mocks["Dataset"].from_pandas.side_effect = [
        MagicMock(name="reference_dataset"),
        MagicMock(name="current_dataset"),
    ]

    return mocks


def test_save_creates_expected_reports(
    sample_data: tuple[pd.DataFrame, pd.DataFrame],
    mocked_dependencies: dict[str, MagicMock],
    tmp_path: Path,
) -> None:
    """Configure Evidently and save HTML/JSON artifacts."""
    reference, current = sample_data
    output_dir = tmp_path / "monitoring" / "reports"

    html_path, json_path = drift_report.DriftReporter().save(
        reference,
        current,
        output_dir,
    )

    assert output_dir.is_dir()
    assert html_path == output_dir / "report.html"
    assert json_path == output_dir / "report.json"

    mocked_dependencies["DataDefinition"].assert_called_once_with(
        numerical_columns=[
            column
            for column in FEATURE_COLUMNS
            if column not in CATEGORICAL_FEATURES
        ],
        categorical_columns=list(CATEGORICAL_FEATURES),
    )

    mocked_dependencies["DataSummaryPreset"].assert_called_once_with()
    mocked_dependencies["DataDriftPreset"].assert_called_once_with(
        columns=list(FEATURE_COLUMNS),
        num_method="wasserstein",
        num_threshold=0.1,
        cat_method="jensenshannon",
        cat_threshold=0.1,
        drift_share=0.5,
    )

    report_factory = mocked_dependencies["Report"]
    report_factory.assert_called_once_with(
        [
            mocked_dependencies["DataSummaryPreset"].return_value,
            mocked_dependencies["DataDriftPreset"].return_value,
        ]
    )

    snapshot = report_factory.return_value.run.return_value
    snapshot.save_html.assert_called_once_with(str(html_path))
    snapshot.save_json.assert_called_once_with(str(json_path))

    mocked_dependencies["save_summary"].assert_called_once_with(
        json_path
    )


def test_save_passes_reference_and_current_correctly(
    sample_data: tuple[pd.DataFrame, pd.DataFrame],
    mocked_dependencies: dict[str, MagicMock],
    tmp_path: Path,
) -> None:
    """Preserve the reference/current distinction in the report."""
    reference, current = sample_data

    drift_report.DriftReporter().save(
        reference,
        current,
        tmp_path / "report",
    )

    processor = mocked_dependencies["DataProcessor"].return_value
    assert processor.select_features.call_count == 2

    pd.testing.assert_frame_equal(
        processor.select_features.call_args_list[0].args[0],
        reference,
    )
    pd.testing.assert_frame_equal(
        processor.select_features.call_args_list[1].args[0],
        current,
    )

    dataset_calls = (
        mocked_dependencies["Dataset"].from_pandas.call_args_list
    )
    assert len(dataset_calls) == 2

    pd.testing.assert_frame_equal(
        dataset_calls[0].args[0],
        reference,
    )
    pd.testing.assert_frame_equal(
        dataset_calls[1].args[0],
        current,
    )

    definition = mocked_dependencies["DataDefinition"].return_value

    for call in dataset_calls:
        assert call.kwargs["data_definition"] is definition

    dataset_results = (
        mocked_dependencies["Dataset"].from_pandas.side_effect
    )
    assert dataset_results is not None

    run_kwargs = (
        mocked_dependencies["Report"].return_value.run
        .call_args.kwargs
    )
    assert "reference_data" in run_kwargs
    assert "current_data" in run_kwargs


def test_save_propagates_report_failure(
    sample_data: tuple[pd.DataFrame, pd.DataFrame],
    mocked_dependencies: dict[str, MagicMock],
    tmp_path: Path,
) -> None:
    """Do not create output artifacts when report generation fails."""
    reference, current = sample_data
    output_dir = tmp_path / "failed_report"

    report = mocked_dependencies["Report"].return_value
    report.run.side_effect = RuntimeError(
        "Simulated Evidently failure"
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated Evidently failure",
    ):
        drift_report.DriftReporter().save(
            reference,
            current,
            output_dir,
        )

    assert not output_dir.exists()
    mocked_dependencies["save_summary"].assert_not_called()
    report.run.return_value.save_html.assert_not_called()
    report.run.return_value.save_json.assert_not_called()
