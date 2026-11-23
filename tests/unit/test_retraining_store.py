"""Test durable retraining reservations using real temporary SQLite files."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from predictive_maintenance.monitoring import retraining_store as store


def claim(
    database: Path,
    dataset_key: str = "raw:v2",
    tracking_uri: str = "http://mlflow.test:5000",
) -> str | None:
    return store.claim_retraining_attempt(
        database_path=database,
        dataset_key=dataset_key,
        tracking_uri=tracking_uri,
        model_type="gradient_boosting",
        context={"window_start": "2026-11-23T00:00:00Z"},
    )


def test_duplicate_input_is_blocked_across_connections(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"

    assert claim(database) is not None
    assert claim(database) is None


def test_new_dataset_can_start_a_new_attempt(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"

    assert claim(database, dataset_key="raw:v2") is not None
    assert claim(database, dataset_key="raw:new-v2") is not None


def test_separate_backend_has_separate_reservation(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"

    assert claim(database) is not None
    assert claim(database, tracking_uri="http://other.test:5000") is not None


@pytest.mark.parametrize("succeeded", [True, False])
def test_finished_attempt_retains_context_and_blocks_duplicates(
    tmp_path: Path,
    succeeded: bool,
) -> None:
    database = tmp_path / "predictions.db"
    attempt_id = claim(database)
    assert attempt_id is not None
    status = "completed" if succeeded else "failed"
    result = {"run_id": "training-run"} if succeeded else {"error": "Training failed."}

    store.finish_retraining_attempt(database, attempt_id, status, result)

    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            """
            SELECT status, finished_at, context, result
            FROM retraining_attempts WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == status
    assert row[1] is not None
    assert json.loads(row[2]) == {"window_start": "2026-11-23T00:00:00Z"}
    assert json.loads(row[3]) == result
    assert claim(database) is None

    with pytest.raises(ValueError, match="missing or already finished"):
        store.finish_retraining_attempt(database, attempt_id, status, result)


def test_unknown_attempt_cannot_be_finished(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"
    assert claim(database) is not None

    with pytest.raises(ValueError, match="missing or already finished"):
        store.finish_retraining_attempt(database, "unknown", "completed", {})


def test_concurrent_claims_allow_only_one_attempt(tmp_path: Path) -> None:
    database = tmp_path / "predictions.db"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, [database, database]))

    assert sum(result is not None for result in results) == 1
