"""Human feedback capture and export (WI-31).

`ai_feedback` is the one AI-DAM table that is NOT derived/rebuildable
state (see schema.py): it records a person's verdict on a review, a single
finding, a similarity/duplicate suggestion, or a face cluster, so it can be
exported later for reviewer tuning or LoRA-style feedback loops. This
module only validates against the same enums as the table's own CHECK
constraints (so a bad value fails with a clear message here rather than a
raw `sqlite3.IntegrityError`) and provides an export path.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional

__all__ = [
    "TARGET_KINDS",
    "VERDICTS",
    "FeedbackValidationError",
    "record_feedback",
    "list_feedback",
    "export_feedback",
]

# What a feedback row passes judgment on; mirrors the ai_feedback.target_kind CHECK.
TARGET_KINDS = ("review", "finding", "similarity", "face_cluster", "duplicate")
# Allowed verdict values; mirrors the ai_feedback.verdict CHECK.
VERDICTS = ("accept", "reject", "false_positive", "rating")

# Every ai_feedback column, in the order rows are listed and exported.
_COLUMNS = (
    "feedback_id",
    "target_kind",
    "target_id",
    "file_id",
    "verdict",
    "rating",
    "note",
    "created_by",
    "created_at",
    "exported_at",
)


class FeedbackValidationError(ValueError):
    """A value would violate one of the `ai_feedback` table's CHECK constraints."""


def record_feedback(
    conn: sqlite3.Connection,
    target_kind: str,
    target_id: str,
    verdict: str,
    file_id: Optional[str] = None,
    rating: Optional[int] = None,
    note: Optional[str] = None,
    created_by: Optional[str] = None,
    now: Optional[float] = None,
) -> int:
    """Insert one feedback row after validating against the schema's enums.

    Validating here (rather than letting SQLite's CHECK constraints raise)
    gives a specific, actionable `FeedbackValidationError` instead of an
    opaque `sqlite3.IntegrityError`.
    """
    if target_kind not in TARGET_KINDS:
        raise FeedbackValidationError(
            f"target_kind must be one of {TARGET_KINDS}, got {target_kind!r}"
        )
    if not target_id:
        raise FeedbackValidationError("target_id is required")
    if verdict not in VERDICTS:
        raise FeedbackValidationError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    if rating is not None:
        if isinstance(rating, bool) or not isinstance(rating, int) or not (1 <= rating <= 5):
            raise FeedbackValidationError(f"rating must be an integer 1..5 or None, got {rating!r}")

    created_at = time.time() if now is None else now
    cur = conn.execute(
        """
        INSERT INTO ai_feedback
            (target_kind, target_id, file_id, verdict, rating, note, created_by,
             created_at, exported_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (target_kind, str(target_id), file_id, verdict, rating, note, created_by, created_at),
    )
    conn.commit()
    return cur.lastrowid


def list_feedback(conn: sqlite3.Connection, unexported_only: bool = False) -> list:
    """All feedback rows as dicts (all columns), ordered by feedback_id."""
    query = f"SELECT {', '.join(_COLUMNS)} FROM ai_feedback"
    if unexported_only:
        query += " WHERE exported_at IS NULL"
    query += " ORDER BY feedback_id"
    rows = conn.execute(query).fetchall()
    return [dict(zip(_COLUMNS, row)) for row in rows]


def export_feedback(
    conn: sqlite3.Connection, out_path: Optional[str] = None, mark: bool = True
) -> str:
    """Export all feedback rows as JSONL (one canonical, sorted-key JSON
    object per row, including every column), optionally writing it to
    `out_path`. When `mark` is True, rows that had never been exported
    (`exported_at IS NULL`) are stamped with the export time first, so the
    returned/written JSONL reflects that stamp; already-exported rows keep
    their original `exported_at`.
    """
    if mark:
        conn.execute(
            "UPDATE ai_feedback SET exported_at = ? WHERE exported_at IS NULL",
            (time.time(),),
        )
        conn.commit()

    rows = list_feedback(conn, unexported_only=False)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text
