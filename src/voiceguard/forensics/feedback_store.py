"""Persistent storage for user detection feedback.

Feedback is intentionally stored separately from the short-lived detection
result store. Every submission starts as unverified and must be independently
verified before it can become training ground truth.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any


_DB_PATH = Path(
    os.environ.get("VG_FEEDBACK_DB", "voiceguard_feedback.db")
).expanduser()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    audio_hash TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    model TEXT NOT NULL,
    feedback TEXT NOT NULL CHECK (feedback IN ('correct', 'incorrect')),
    user TEXT NOT NULL,
    timestamp REAL NOT NULL,
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    verified_label TEXT,
    verified_by TEXT,
    verified_at REAL
);

CREATE INDEX IF NOT EXISTS idx_feedback_audio_hash
    ON feedback(audio_hash);

CREATE INDEX IF NOT EXISTS idx_feedback_verification
    ON feedback(verification_status);
"""


class FeedbackStore:
    """SQLite-backed persistent store for detection feedback."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_db(self) -> None:
        parent = self.db_path.parent

        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)

    def record(
        self,
        audio_hash: str,
        predicted_label: str,
        predicted_confidence: float,
        model: str,
        feedback: str,
        user: str,
        timestamp: float,
    ) -> dict[str, Any]:
        """Store new feedback as unverified."""

        feedback_id = str(uuid.uuid4())

        record = {
            "feedback_id": feedback_id,
            "audio_hash": audio_hash,
            "predicted_label": predicted_label,
            "predicted_confidence": float(predicted_confidence),
            "model": model,
            "feedback": feedback,
            "user": user,
            "timestamp": float(timestamp),
            "verification_status": "unverified",
            "verified_label": None,
            "verified_by": None,
            "verified_at": None,
        }

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id,
                        audio_hash,
                        predicted_label,
                        predicted_confidence,
                        model,
                        feedback,
                        user,
                        timestamp,
                        verification_status,
                        verified_label,
                        verified_by,
                        verified_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["feedback_id"],
                        record["audio_hash"],
                        record["predicted_label"],
                        record["predicted_confidence"],
                        record["model"],
                        record["feedback"],
                        record["user"],
                        record["timestamp"],
                        record["verification_status"],
                        record["verified_label"],
                        record["verified_by"],
                        record["verified_at"],
                    ),
                )

        return record

    def get(self, feedback_id: str) -> dict[str, Any] | None:
        """Return one feedback record by ID."""

        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        feedback_id,
                        audio_hash,
                        predicted_label,
                        predicted_confidence,
                        model,
                        feedback,
                        user,
                        timestamp,
                        verification_status,
                        verified_label,
                        verified_by,
                        verified_at
                    FROM feedback
                    WHERE feedback_id = ?
                    """,
                    (feedback_id,),
                ).fetchone()

        if row is None:
            return None

        return dict(row)

    def list_unverified(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return feedback waiting for independent verification."""

        limit = max(1, min(int(limit), 1000))

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        feedback_id,
                        audio_hash,
                        predicted_label,
                        predicted_confidence,
                        model,
                        feedback,
                        user,
                        timestamp,
                        verification_status,
                        verified_label,
                        verified_by,
                        verified_at
                    FROM feedback
                    WHERE verification_status = 'unverified'
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]

    def verify(
        self,
        feedback_id: str,
        verified_label: str,
        verified_by: str,
        verified_at: float,
    ) -> dict[str, Any] | None:
        """Mark feedback as independently verified.

        The verified label becomes the future training ground-truth label.
        """

        verified_label = verified_label.strip().lower()

        if verified_label not in {"real", "fake"}:
            raise ValueError(
                "verified_label must be 'real' or 'fake'."
            )

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE feedback
                    SET
                        verification_status = 'verified',
                        verified_label = ?,
                        verified_by = ?,
                        verified_at = ?
                    WHERE
                        feedback_id = ?
                        AND verification_status = 'unverified'
                    """,
                    (
                        verified_label,
                        verified_by,
                        float(verified_at),
                        feedback_id,
                    ),
                )

                if cursor.rowcount == 0:
                    return None

        return self.get(feedback_id)

    def list_verified(
        self,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return independently verified feedback records."""

        limit = max(1, min(int(limit), 5000))

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        feedback_id,
                        audio_hash,
                        predicted_label,
                        predicted_confidence,
                        model,
                        feedback,
                        user,
                        timestamp,
                        verification_status,
                        verified_label,
                        verified_by,
                        verified_at
                    FROM feedback
                    WHERE verification_status = 'verified'
                    ORDER BY verified_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]


# Module-level store used by the API.
_store = FeedbackStore()


def record(
    audio_hash: str,
    predicted_label: str,
    predicted_confidence: float,
    model: str,
    feedback: str,
    user: str,
    timestamp: float,
) -> dict[str, Any]:
    return _store.record(
        audio_hash=audio_hash,
        predicted_label=predicted_label,
        predicted_confidence=predicted_confidence,
        model=model,
        feedback=feedback,
        user=user,
        timestamp=timestamp,
    )


def get(feedback_id: str) -> dict[str, Any] | None:
    return _store.get(feedback_id)


def list_unverified(limit: int = 100) -> list[dict[str, Any]]:
    return _store.list_unverified(limit)


def verify(
    feedback_id: str,
    verified_label: str,
    verified_by: str,
    verified_at: float,
) -> dict[str, Any] | None:
    return _store.verify(
        feedback_id=feedback_id,
        verified_label=verified_label,
        verified_by=verified_by,
        verified_at=verified_at,
    )


def list_verified(limit: int = 1000) -> list[dict[str, Any]]:
    return _store.list_verified(limit)