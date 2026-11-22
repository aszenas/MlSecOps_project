"""Tests for the reproducible preprocessed submission export."""

import json
from pathlib import Path

import pandas as pd

from predictive_maintenance.data.submission_export import (
    export_preprocessed_submission,
)


def test_export_uses_one_train_only_fit_and_stable_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The export fits once on train and transforms all four partitions."""
    raw = pd.DataFrame(
        {
            "UDI": range(1, 5001),
            "Product ID": [f"L{i}" for i in range(5000)],
            "Type": ["L", "M", "H", "L", "M"] * 1000,
            "Air temperature [K]": [298.0 + i / 100 for i in range(5000)],
            "Process temperature [K]": [308.0 + i / 100 for i in range(5000)],
            "Rotational speed [rpm]": [1400 + i for i in range(5000)],
            "Torque [Nm]": [30.0 + i / 10 for i in range(5000)],
            "Tool wear [min]": list(range(5000)),
            "Target": [0, 1] * 2500,
            "Failure Type": ["No Failure"] * 5000,
        }
    )
    monkeypatch.setattr(
        "predictive_maintenance.data.submission_export.verify_dvc_dataset",
        lambda root: "a" * 32 + ".dir",
    )
    monkeypatch.setattr(
        "predictive_maintenance.data.submission_export.load_raw_data",
        lambda path: raw.copy(deep=True),
    )

    from predictive_maintenance.data import submission_export

    original_factory = submission_export.DataProcessor.create_preprocessing_pipeline
    calls = {"fit": 0, "transform": 0}

    class CountingPipeline:
        def __init__(self):
            self.pipeline = original_factory(DataProcessor())

        def fit(self, frame):
            calls["fit"] += 1
            self.pipeline.fit(frame)
            return self

        def transform(self, frame):
            calls["transform"] += 1
            return self.pipeline.transform(frame)

    from predictive_maintenance.data.preprocessing import DataProcessor

    monkeypatch.setattr(
        submission_export.DataProcessor,
        "create_preprocessing_pipeline",
        lambda self: CountingPipeline(),
    )

    first = export_preprocessed_submission(tmp_path, tmp_path / "first")
    second = export_preprocessed_submission(tmp_path, tmp_path / "second")

    assert calls == {"fit": 2, "transform": 8}
    assert first.partition_counts == {
        "train": 3000,
        "validation": 750,
        "future": 750,
        "test": 500,
    }
    assert first.feature_names == second.feature_names
    assert first.csv_path.read_bytes() == second.csv_path.read_bytes()
    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()

    exported = pd.read_csv(first.csv_path)
    assert len(exported) == 5000
    assert list(exported.columns) == [
        *first.feature_names,
        "target",
        "partition",
        "source_udi",
    ]
    assert sorted(exported["source_udi"].tolist()) == list(range(1, 5001))
    assert json.loads(first.metadata_path.read_text(encoding="utf-8"))["preprocessing"][
        "fit_partition"
    ] == "train"
    assert json.loads(first.metadata_path.read_text(encoding="utf-8"))["split_policy"][
        "version"
    ] == "stratified-v1-seed42-train3000-validation750-future750-test500"
