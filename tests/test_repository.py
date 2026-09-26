from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.backend_metier_api.app.repository import (
    get_prediction,
    init_database,
    save_feedback,
    save_prediction,
)


def test_save_and_get_prediction(tmp_path):
    database_path = tmp_path / "backend_metier.db"
    init_database(database_path)

    created_at = datetime.now(timezone.utc)

    save_prediction(
        database_path,
        request_id="request-123",
        input_data={
            "age": 35,
            "synthese_entretien": "Recherche active.",
        },
        prediction=1,
        probabilities={0: 0.2, 1: 0.6, 2: 0.2},
        decision="Retour à l'emploi intermédiaire",
        model_version="v1",
        created_at=created_at,
    )

    prediction = get_prediction(database_path, "request-123")

    assert prediction is not None
    assert prediction["request_id"] == "request-123"
    assert prediction["prediction"] == 1
    assert prediction["input_data"]["age"] == 35
    assert prediction["model_version"] == "v1"

    # Les clés d'un dictionnaire deviennent des chaînes après JSON.
    assert prediction["probabilities"] == {
        "0": 0.2,
        "1": 0.6,
        "2": 0.2,
    }


def test_get_prediction_returns_none_for_unknown_request(tmp_path):
    database_path = tmp_path / "backend_metier.db"
    init_database(database_path)

    assert get_prediction(database_path, "unknown-request") is None


def test_save_feedback_links_to_existing_prediction(tmp_path):
    database_path = tmp_path / "backend_metier.db"
    init_database(database_path)

    save_prediction(
        database_path,
        request_id="request-456",
        input_data={"age": 42},
        prediction=2,
        probabilities={0: 0.1, 1: 0.2, 2: 0.7},
        decision="Risque de chômage de longue durée",
        model_version="v1",
        created_at=datetime.now(timezone.utc),
    )

    feedback_id, created_at = save_feedback(
        database_path,
        request_id="request-456",
        corrected_class=1,
        comment="La situation s'est améliorée après une formation.",
    )

    assert feedback_id == 1
    assert created_at.tzinfo is not None


def test_save_feedback_rejects_unknown_prediction(tmp_path):
    database_path = tmp_path / "backend_metier.db"
    init_database(database_path)

    with pytest.raises(LookupError, match="Prédiction inconnue"):
        save_feedback(
            database_path,
            request_id="unknown-request",
            corrected_class=1,
            comment=None,
        )