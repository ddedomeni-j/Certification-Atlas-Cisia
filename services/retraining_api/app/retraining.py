from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import (BACKEND_DB_PATH,
                    CANDIDATE_VERSION,
                    DATA_DIR,
                    PROJECT_ROOT,
                    REPORTS_DIR,
                    TARGET_COLUMN,
                    TRAINING_DATA_PATH)

from services.backend_metier_api.app.repository import (list_feedbacks_for_training,
                                                        mark_feedbacks_as_used_for_training)

RETRAINING_DATA_PATH = DATA_DIR / "training_set_with_feedback.csv"


def run_script(script_name: str, *arguments: str) -> None:
    completed = subprocess.run([sys.executable,
                                str(PROJECT_ROOT / "Scripts" / script_name),
                                *arguments,
                               ],
                               cwd=PROJECT_ROOT,
                               capture_output=True,
                               text=True,
                               check=False)

    if completed.returncode != 0:
        raise RuntimeError(f"{script_name} a échoué : "
                           f"{completed.stderr or completed.stdout}")


def build_enriched_training_set(
    feedbacks: list[dict[str, object]],
) -> Path:
    """Ajoute les corrections conseiller au jeu d'entraînement initial."""
    training_data = pd.read_csv(
        TRAINING_DATA_PATH,
        dtype={
            "code_insee_commune": "string",
            "code_rome_vise": "string",
        },
    )

    feedback_rows = []

    for feedback in feedbacks:
        input_data = dict(feedback["input_data"])
        input_data[TARGET_COLUMN] = feedback["corrected_class"]
        feedback_rows.append(input_data)

    feedback_data = pd.DataFrame(feedback_rows)

    # Conserve exactement les colonnes et leur ordre du jeu d'origine.
    feedback_data = feedback_data.reindex(columns=training_data.columns)

    enriched_data = pd.concat(
        [training_data, feedback_data],
        ignore_index=True,
    )

    enriched_data.to_csv(RETRAINING_DATA_PATH, index=False)
    return RETRAINING_DATA_PATH


def run_retraining() -> dict[str, str | int]:
    feedbacks = list_feedbacks_for_training(BACKEND_DB_PATH)

    if not feedbacks:
        raise ValueError("Aucun feedback disponible pour le réentraînement.")

    feedback_ids = [feedback["feedback_id"] for feedback in feedbacks]
    feedback_count = len(feedbacks)

    enriched_training_path = build_enriched_training_set(feedbacks)

    run_script("train_and_persist.py", "--data-path", str(enriched_training_path))

    # Évaluer le candidat face au modèle actif.
    run_script("evaluate_candidate.py")

    # Générer une recommandation, sans promotion automatique.
    run_script("promotion.py")

    decision_path = REPORTS_DIR / "promotion_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    # Seulement après réussite complète du cycle.
    mark_feedbacks_as_used_for_training(BACKEND_DB_PATH, feedback_ids)

    return {
        "status": "completed",
        "feedback_count": feedback_count,
        "candidate_version": CANDIDATE_VERSION,
        "recommendation": decision["reason"],
    }