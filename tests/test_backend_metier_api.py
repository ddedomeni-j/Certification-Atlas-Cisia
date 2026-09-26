import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.backend_metier_api.app import main


VALID_APPLICATION = {
    "age": 35,
    "anciennete_poste_ans": 2.5,
    "niveau_diplome": "Bac+2",
    "est_allocataire": 1.0,
    "nationalite_hors_ue": 0,
    "departement": "75",
    "domaine_rome": "M",
    "synthese_entretien": "Compétences à réactualiser.",
}


def test_health_returns_ok():
    client = TestClient(main.app)

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_business_decision(monkeypatch, tmp_path):
    async def fake_predict_with_model(
        application: dict,
        request_id: str,
    ) -> dict:
        assert application == VALID_APPLICATION

        return {
            "request_id": request_id,
            "prediction": 2,
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            "model_version": "v1",
        }

    monkeypatch.setattr(main, "predict_with_model", fake_predict_with_model)
    monkeypatch.setattr(main, "BACKEND_DB_PATH", tmp_path / "backend_metier.db")

    with TestClient(main.app) as client:
        response = client.post(
            "/applications/predict",
            json=VALID_APPLICATION,
        )

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == 2
    assert data["decision"] == "Risque de chômage de longue durée"
    assert data["model_version"] == "v1"
    assert data["probabilities"]["2"] == 0.7
    assert data["request_id"]


def test_predict_rejects_invalid_application():
    invalid_application = {**VALID_APPLICATION, "age": -1}

    client = TestClient(main.app)
    response = client.post("/applications/predict", json=invalid_application)

    assert response.status_code == 422