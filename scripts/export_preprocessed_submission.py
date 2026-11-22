"""Export the reproducibly preprocessed submission dataset."""

import argparse
import json
from pathlib import Path

from predictive_maintenance.data.submission_export import (
    export_preprocessed_submission,
)


def main() -> None:
    """Parse export paths and write the submission artifacts."""
    parser = argparse.ArgumentParser(
        description="Export the train-fitted preprocessed submission dataset.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing data/raw.dvc and the raw dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to data/processed/submission.",
    )
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    output_dir = args.output_dir or repository_root / "data/processed/submission"
    result = export_preprocessed_submission(repository_root, output_dir)
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
