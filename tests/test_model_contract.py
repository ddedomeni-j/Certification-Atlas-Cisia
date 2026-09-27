import pytest

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))

from config import (
    CATEGORICAL_FEATURES,
    MODEL_VERSION,
    MODELS_DIR,
    NUMERIC_FEATURES,
)

def test_reloaded_s5_model_predicts_from_raw_features():
    model_path = MODELS_DIR / f"pipeline_{MODEL_VERSION}.joblib"
    metadata_path = MODELS_DIR / f"pipeline_{MODEL_VERSION}.json"

    assert model_path.exists(), (
        "Artefact absent : exécutez d'abord "
        "`python Scripts/train_and_persist.py`."
    )
    assert metadata_path.exists()

    model = joblib.load(model_path)

    application = pd.DataFrame(
        [
            {
                "age": 35,
                "anciennete_poste_ans": 2.5,
                "niveau_diplome": "Bac+2",
                "est_allocataire": 1.0,
                "nationalite_hors_ue": 0,
                "departement": "75",
                "code_rome_vise": "M1805",
                "synthese_entretien": (
                    "Compétences à réactualiser sur les outils numériques."
                ),
            }
        ]
    )

    expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert list(application.columns) == expected_features

    prediction = model.predict(application)
    probabilities = model.predict_proba(application)

    assert prediction.shape == (1,)
    assert prediction[0] in {0, 1, 2}

    assert probabilities.shape == (1, 3)
    assert probabilities[0].sum() == pytest.approx(1.0)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == expected_features