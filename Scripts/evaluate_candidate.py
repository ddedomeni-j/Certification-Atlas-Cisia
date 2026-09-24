from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from config import (
    CANDIDATE_VERSION,
    MODEL_VERSION,
    MODELS_DIR,
    REFERENCE_SET_PATH,
    REPORTS_DIR,
)
from data_preparation import load_dataset
from evaluate import evaluate_model


def evaluate_artifact(version: str) -> dict[str, Any]:
    """Évalue une version d'artefact sur le jeu de référence figé."""
    model_path = MODELS_DIR / f"pipeline_{version}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Artefact absent : {model_path}")

    X_reference, y_reference = load_dataset(REFERENCE_SET_PATH)
    model = joblib.load(model_path)

    return {
        "model_version": version,
        "reference_set": REFERENCE_SET_PATH.name,
        "metrics": evaluate_model(model, X_reference, y_reference),
    }


def main() -> None:
    """Compare le candidat et la production sur le même jeu de référence."""
    production = evaluate_artifact(MODEL_VERSION)
    candidate = evaluate_artifact(CANDIDATE_VERSION)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "candidate_evaluation.json"
    report_path.write_text(
        json.dumps(
            {"production": production, "candidate": candidate},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Rapport créé : {report_path}")
    print(f"F1 production : {production['metrics']['f1_macro']:.3f}")
    print(f"F1 candidat : {candidate['metrics']['f1_macro']:.3f}")


if __name__ == "__main__":
    main()