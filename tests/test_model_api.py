import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.model_api.app.main import app

from fastapi.testclient import TestClient

from services.model_api.app.main import app

VALID_APPLICATION = {
    "age": 35,
    "anciennete_poste_ans": 2.5,
    "niveau_diplome": "Bac+2",
    "est_allocataire": 1.0,
    "nationalite_hors_ue": 0,
    "departement": "75",
    "domaine_rome": "M",
    "synthese_entretien": (
        "Compétences à réactualiser sur les outils numériques."
    ),
}


def test_health_returns_ok_when_model_is_loaded():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers


def test_info_returns_active_s5_model_metadata():
    with TestClient(app) as client:
        response = client.get("/info")

    assert response.status_code == 200

    data = response.json()
    assert data["model_name"] == "S5_LightGBM"
    assert data["model_version"] == "v1"
    assert data["scenario"] == "S5"
    assert len(data["feature_columns"]) == 8


def test_predict_returns_class_probabilities_and_request_id():
    request_id = "test-request-123"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=VALID_APPLICATION,
            headers={"X-Request-ID": request_id},
        )

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in {0, 1, 2}
    assert set(data["probabilities"]) == {"0", "1", "2"}
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=0.001)
    assert data["model_version"] == "v1"
    assert data["request_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id


def test_predict_rejects_invalid_schema():
    invalid_application = {**VALID_APPLICATION, "age": -1}

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_application)

    assert response.status_code == 422