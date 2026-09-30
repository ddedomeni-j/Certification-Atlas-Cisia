from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database(database_path: Path) -> None:
    """Crée les tables SQLite nécessaires au backend métier."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                input_data_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probabilities_json TEXT NOT NULL,
                decision TEXT NOT NULL,
                review_status TEXT NOT NULL,
                model_version TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                corrected_class INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (request_id) REFERENCES predictions(request_id)
            );
            """
        )


def save_prediction(
    database_path: Path,
    *,
    request_id: str,
    input_data: dict[str, Any],
    prediction: int,
    probabilities: dict[str | int, float],
    decision: str,
    model_version: str,
    review_status: str,
    created_at: datetime,
) -> None:
    """Enregistre une inférence et les données métier associées."""
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                request_id,
                created_at,
                input_data_json,
                prediction,
                probabilities_json,
                decision,
                review_status,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                created_at.isoformat(),
                json.dumps(input_data, ensure_ascii=False),
                prediction,
                json.dumps(probabilities),
                decision,
                review_status,
                model_version,
            ),
        )


def get_prediction(
    database_path: Path,
    request_id: str,
) -> dict[str, Any] | None:
    """Retourne une prédiction stockée, ou None si elle est inconnue."""
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE request_id = ?",
            (request_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "request_id": row["request_id"],
        "created_at": row["created_at"],
        "input_data": json.loads(row["input_data_json"]),
        "prediction": row["prediction"],
        "probabilities": json.loads(row["probabilities_json"]),
        "decision": row["decision"],
        "review_status": row["review_status"],
        "model_version": row["model_version"],
    }


def save_feedback(
    database_path: Path,
    *,
    request_id: str,
    corrected_class: int,
    comment: str | None,
) -> tuple[int, datetime]:
    """Enregistre une correction liée à une prédiction existante."""
    created_at = datetime.now(timezone.utc)

    with _connect(database_path) as connection:
        prediction_exists = connection.execute(
            "SELECT 1 FROM predictions WHERE request_id = ?",
            (request_id,),
        ).fetchone()

        if prediction_exists is None:
            raise LookupError(f"Prédiction inconnue : {request_id}")

        cursor = connection.execute(
            """
            INSERT INTO feedbacks (
                request_id,
                corrected_class,
                comment,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                request_id,
                corrected_class,
                comment,
                created_at.isoformat(),
            ),
        )

    return int(cursor.lastrowid), created_at

def list_predictions(
    database_path: Path,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Liste les prédictions récentes, de la plus récente à la plus ancienne."""
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                request_id,
                created_at,
                prediction,
                decision,
                review_status,
                model_version
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]

def list_feedbacks_for_training(database_path: Path) -> list[dict[str, Any]]:
    """Retourne les feedbacks non utilisés avec le dossier initial associé."""
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                feedbacks.id AS feedback_id,
                feedbacks.request_id,
                feedbacks.corrected_class,
                feedbacks.comment,
                feedbacks.created_at AS feedback_created_at,
                predictions.input_data_json,
                predictions.model_version AS predicted_with_version
            FROM feedbacks
            INNER JOIN predictions
                ON predictions.request_id = feedbacks.request_id
            WHERE feedbacks.used_for_training = 0
            ORDER BY feedbacks.created_at ASC
            """
        ).fetchall()

    return [
        {
            "feedback_id": row["feedback_id"],
            "request_id": row["request_id"],
            "input_data": json.loads(row["input_data_json"]),
            "corrected_class": row["corrected_class"],
            "comment": row["comment"],
            "feedback_created_at": row["feedback_created_at"],
            "predicted_with_version": row["predicted_with_version"],
        }
        for row in rows
    ]


def mark_feedbacks_as_used_for_training(
    database_path: Path,
    feedback_ids: list[int],
) -> None:
    """Marque les feedbacks intégrés avec succès dans un entraînement candidat."""
    if not feedback_ids:
        return

    placeholders = ", ".join("?" for _ in feedback_ids)

    with _connect(database_path) as connection:
        connection.execute(
            f"""
            UPDATE feedbacks
            SET used_for_training = 1
            WHERE id IN ({placeholders})
            """,
            feedback_ids,
        )