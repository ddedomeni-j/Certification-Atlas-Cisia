import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))

from config import (
    CATEGORICAL_FEATURES,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_SCENARIO,
    NUMERIC_FEATURES,
)

from persist import save_model

def test_save_model_creates_reloadable_artifact_and_metadata(tmp_path):
    version = MODEL_VERSION
    
    X_train = pd.DataFrame({"synthese_entretien": ["texte un", "texte deux"]})
    y_train = [0, 1]

    pipeline = Pipeline(
        steps=[("model", DummyClassifier(strategy="most_frequent"))]
    )
    pipeline.fit(X_train, y_train)

    model_path, metadata_path = save_model(
        pipeline=pipeline,
        output_dir=tmp_path,
        version=version,
        metrics={"f1_macro_holdout": 0.65, "critical_errors_2_to_0": 0},
        best_params={"model__n_estimators": 100},
        train_rows=80,
        holdout_rows=20,
    )

    assert model_path.exists()
    assert metadata_path.exists()

    reloaded_pipeline = joblib.load(model_path)
    assert len(reloaded_pipeline.predict(X_train)) == 2

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert model_path.name == f"pipeline_{version}.joblib"
    assert metadata_path.name == f"pipeline_{version}.json"
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["artifact"] == f"pipeline_{version}.joblib"
    assert metadata["model_name"] == MODEL_NAME
    assert metadata["scenario"] == MODEL_SCENARIO
    assert metadata["feature_columns"] == NUMERIC_FEATURES + CATEGORICAL_FEATURES