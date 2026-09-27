import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.retraining_api.app import main


def test_retraining_returns_candidate_recommendation(monkeypatch):
    monkeypatch.setattr(main,
                        "run_retraining",
                        lambda: {
                            "status": "completed",
                            "feedback_count": 3,
                            "candidate_version": "candidate",
                            "recommendation": "Promotion recommandée.",
                        })

    with TestClient(main.app) as client:
        response = client.post("/retraining")

    assert response.status_code == 201
    assert response.json()["feedback_count"] == 3
    assert response.json()["recommendation"] == "Promotion recommandée."


def test_retraining_returns_422_when_no_feedback_is_available(monkeypatch):
    def no_feedback() -> dict:
        raise ValueError("Aucun feedback disponible pour le réentraînement.")

    monkeypatch.setattr(main, "run_retraining", no_feedback)

    with TestClient(main.app) as client:
        response = client.post("/retraining")

    assert response.status_code == 422
    assert "Aucun feedback" in response.json()["detail"]


def test_retraining_returns_500_when_training_fails(monkeypatch):
    def failed_training() -> dict:
        raise RuntimeError("Erreur d'entraînement")

    monkeypatch.setattr(main, "run_retraining", failed_training)

    with TestClient(main.app) as client:
        response = client.post("/retraining")

    assert response.status_code == 500
    assert response.json()["detail"] == "Échec du réentraînement : RuntimeError"