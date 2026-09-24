import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))

from evaluate import evaluate_model


def test_evaluate_model_returns_expected_metrics():
    X = pd.DataFrame(
        {
            "synthese_entretien": [
                "profil autonome",
                "freins mobilité",
                "reconversion",
                "recherche active",
            ]
        }
    )
    y = pd.Series([0, 1, 2, 2])

    pipeline = Pipeline(
        steps=[("model", DummyClassifier(strategy="constant", constant=0))]
    )
    pipeline.fit(X, y)

    metrics = evaluate_model(pipeline, X, y)

    assert metrics["rows_evaluated"] == 4
    assert metrics["accuracy"] == 0.25
    assert metrics["f1_macro"] >= 0.0
    assert metrics["critical_errors_2_to_0"] == 2
    assert metrics["critical_error_rate_2_to_0"] == 1.0
    assert metrics["confusion_matrix"] == [
        [1, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ]
    assert "0" in metrics["classification_report"]
    assert "macro avg" in metrics["classification_report"]