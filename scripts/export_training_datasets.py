"""Export the v1 and v2 training subsets from the raw dataset."""

from pathlib import Path
from typing import cast

import pandas as pd

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.data.splitting import (
    create_initial_data_splits,
    create_retraining_data_splits,
)


def main() -> None:
    """Write the v1 and v2 training rows in the original CSV schema."""
    root = Path(__file__).resolve().parents[1]
    raw = load_raw_data(root / "data/raw/predictive_maintenance_5000.csv")

    processor = DataProcessor()
    features, target = processor.split_features_target(processor.rename_columns(raw))
    v1 = create_initial_data_splits(features, target)
    source_udis = cast(pd.Series, raw["UDI"])
    v2 = create_retraining_data_splits(v1, source_udis)

    output_dir = root / "data/processed/training"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, splits in (("train_v1.csv", v1), ("train_v2.csv", v2)):
        training_rows = raw.loc[splits.X_train.index]
        training_rows.to_csv(output_dir / filename, index=False, lineterminator="\n")
        print(f"{filename}: {len(training_rows)} rows")


if __name__ == "__main__":
    main()
