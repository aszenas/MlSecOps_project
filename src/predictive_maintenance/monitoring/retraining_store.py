"""Persist retraining attempts and prevent duplicate training inputs."""

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from predictive_maintenance.models.training import ModelType


def claim_retraining_attempt(
    database_path: Path,
    dataset_key: str,
    tracking_uri: str,
    model_type: ModelType,
    context: Mapping[str, object],
) -> str | None:
    """Atomically reserve one dataset/backend/model combination."""
    input_key = json.dumps([tracking_uri, model_type, dataset_key])
    attempt_id = str(uuid4())
    payload = json.dumps(dict(context), allow_nan=False)
    started_at = datetime.now(timezone.utc).isoformat()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path, timeout=10)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retraining_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    input_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL
                        CHECK (status IN ('running', 'completed', 'failed')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    context TEXT NOT NULL,
                    result TEXT
                )
                """
            )
            cursor = connection.execute(
                """
                INSERT INTO retraining_attempts (
                    attempt_id, input_key, status, started_at, context
                )
                VALUES (?, ?, 'running', ?, ?)
                ON CONFLICT(input_key) DO NOTHING
                """,
                (attempt_id, input_key, started_at, payload),
            )
            return attempt_id if cursor.rowcount == 1 else None


def finish_retraining_attempt(
    database_path: Path,
    attempt_id: str,
    status: Literal["completed", "failed"],
    result: Mapping[str, object],
) -> None:
    """Finish a running attempt without replacing its original context."""
    payload = json.dumps(dict(result), allow_nan=False)
    finished_at = datetime.now(timezone.utc).isoformat()

    with closing(sqlite3.connect(database_path, timeout=10)) as connection:
        with connection:
            cursor = connection.execute(
                """
                UPDATE retraining_attempts
                SET status = ?, finished_at = ?, result = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (status, finished_at, payload, attempt_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("The retraining attempt is missing or already finished.")
