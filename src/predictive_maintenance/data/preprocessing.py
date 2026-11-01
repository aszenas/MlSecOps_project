
"""Data preparation and feature selection for predictive maintenance."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


COLUMN_MAPPING = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Target": "target",
    "Failure Type": "failure_type",
}

FEATURE_COLUMNS = (
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)

TARGET_COLUMN = "target"

NUMERIC_FEATURES = (
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temperature_difference_k",
)

CATEGORICAL_FEATURES = ("type",)

ENGINEERED_FEATURES = (
    "temperature_difference_k",
)

class DataProcessor:
    """Prepare predictive maintenance data for classification."""

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename raw dataset columns to consistent snake_case names."""
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        renamed_df = df.rename(columns=COLUMN_MAPPING)

        if not renamed_df.columns.is_unique:
            raise ValueError("Duplicate column names after renaming.")

        return renamed_df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select model input features in a consistent column order."""
        missing_columns = [
            column for column in FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required feature columns: {missing_columns}"
            )

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        return df.loc[:, list(FEATURE_COLUMNS)].copy()

    def split_features_target(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate model features (X) and binary target (y)."""
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Missing target column: {TARGET_COLUMN}"
            )

        target = df[TARGET_COLUMN]

        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError(
            )

        features = self.select_features(df)

        return features, target.copy()
    
    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temperature difference without modifying input data."""
        required = (
            "air_temperature_k",
            "process_temperature_k",
        )

        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(
                f"Missing columns for feature engineering: {missing}"
            )

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        result = df.copy()

        result["temperature_difference_k"] = (
            result["process_temperature_k"]
            - result["air_temperature_k"]
        )

        return result

    
    def create_imputer(self) -> ColumnTransformer:
        """Create an unfitted transformer for missing value imputation."""
        numeric_imputer = SimpleImputer(
            strategy="median",
        )

        categorical_imputer = SimpleImputer(
            strategy="most_frequent",
        )

        transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_imputer, list(NUMERIC_FEATURES)),
                ("categorical", categorical_imputer, list(CATEGORICAL_FEATURES)),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        return transformer.set_output(transform="pandas")


