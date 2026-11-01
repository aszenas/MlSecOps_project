
"""Unit tests for dataset loading and raw schema validation."""

import pandas as pd
import pytest

from src.predictive_maintenance.data.loader import (
    load_raw_data,
    validate_raw_data,
)


@pytest.fixture
def valid_raw_data() -> pd.DataFrame:
    """Create a minimal valid predictive maintenance dataset."""
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3],
            "Product ID": ["L00001", "M00002", "H00003"],
            "Type": ["L", "M", "H"],
            "Air temperature [K]": [298.1, 299.2, 300.3],
            "Process temperature [K]": [308.1, 309.2, 310.3],
            "Rotational speed [rpm]": [1500, 1600, 1700],
            "Torque [Nm]": [40.1, 42.2, 39.3],
            "Tool wear [min]": [10, 20, 30],
            "Target": [0, 1, 0],
            "Failure Type": [
                "No Failure",
                "Power Failure",
                "No Failure",
            ],
        }
    )


def test_load_valid_csv(
    tmp_path,
    valid_raw_data: pd.DataFrame,
) -> None:
    """Valid CSV should load with unchanged content."""
    csv_path = tmp_path / "valid.csv"
    valid_raw_data.to_csv(csv_path, index=False)

    result = load_raw_data(csv_path)

    pd.testing.assert_frame_equal(result, valid_raw_data)


def test_missing_file() -> None:
    """Missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_raw_data("nonexistent_dataset.csv")


def test_empty_dataset(tmp_path, valid_raw_data: pd.DataFrame) -> None:
    """CSV containing only headers should be rejected."""
    csv_path = tmp_path / "empty.csv"
    valid_raw_data.iloc[:0].to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="dataset is empty"):
        load_raw_data(csv_path)


def test_missing_required_column(valid_raw_data: pd.DataFrame) -> None:
    """Missing required columns should be reported."""
    invalid = valid_raw_data.drop(columns=["Torque [Nm]"])

    with pytest.raises(ValueError, match="Torque"):
        validate_raw_data(invalid)


def test_duplicate_csv_headers(
    tmp_path,
    valid_raw_data: pd.DataFrame,
) -> None:
    """Duplicate column names in a CSV header should be rejected."""
    csv_path = tmp_path / "duplicate.csv"
    valid_raw_data.to_csv(csv_path, index=False)

    content = csv_path.read_text(encoding="utf-8")
    content = content.replace(
        "UDI,Product ID",
        "UDI,UDI",
        1,
    )
    csv_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_raw_data(csv_path)


def test_duplicate_dataframe_columns(valid_raw_data: pd.DataFrame) -> None:
    """Duplicate DataFrame column names should be rejected."""
    invalid = valid_raw_data.copy()
    invalid.columns = [
        "UDI" if column == "Product ID" else column
        for column in invalid.columns
    ]

    with pytest.raises(ValueError, match="duplicate column names"):
        validate_raw_data(invalid)


@pytest.mark.parametrize(
    "column,invalid_value",
    [
        ("Air temperature [K]", "invalid"),
        ("Torque [Nm]", "invalid"),
        ("Rotational speed [rpm]", "invalid"),
    ],
)
def test_invalid_numeric_type(
    valid_raw_data: pd.DataFrame,
    column: str,
    invalid_value: str,
) -> None:
    """Invalid numeric values should be rejected."""
    invalid = valid_raw_data.copy()
    invalid[column] = invalid_value

    with pytest.raises(ValueError, match="must be numeric"):
        validate_raw_data(invalid)


def test_non_integer_value(valid_raw_data: pd.DataFrame) -> None:
    """Integer-only columns should reject fractional values."""
    invalid = valid_raw_data.copy()
    invalid["Tool wear [min]"] = [10.5, 20.0, 30.0]

    with pytest.raises(ValueError, match="must contain integers"):
        validate_raw_data(invalid)


@pytest.mark.parametrize("invalid_target", [2, -1, None])
def test_invalid_target(
    valid_raw_data: pd.DataFrame,
    invalid_target: int | None,
) -> None:
    """Target must contain only 0 and 1."""
    invalid = valid_raw_data.copy()
    invalid.loc[0, "Target"] = invalid_target

    with pytest.raises(ValueError):
        validate_raw_data(invalid)


def test_invalid_product_type(valid_raw_data: pd.DataFrame) -> None:
    """Unexpected product categories should be rejected."""
    invalid = valid_raw_data.copy()
    invalid.loc[0, "Type"] = "X"

    with pytest.raises(ValueError, match="Unexpected product types"):
        validate_raw_data(invalid)
