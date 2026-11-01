import pandas as pd


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


class DataProcessor:

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        renamed_df = df.rename(columns=COLUMN_MAPPING)

        if not renamed_df.columns.is_unique:
            raise ValueError("Duplicate column names after renaming.")

        return renamed_df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
