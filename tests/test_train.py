import sys
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))

from config import LGBM_PARAMS
from train import build_pipeline


def test_build_pipeline_uses_s5_lightgbm_configuration():
    pipeline = build_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    assert isinstance(pipeline.named_steps["model"], LGBMClassifier)

    model_params = pipeline.named_steps["model"].get_params()
    assert model_params["objective"] == LGBM_PARAMS["objective"]
    assert model_params["n_estimators"] == LGBM_PARAMS["n_estimators"]
    assert model_params["learning_rate"] == LGBM_PARAMS["learning_rate"]
    assert model_params["num_leaves"] == LGBM_PARAMS["num_leaves"]
    assert model_params["min_child_samples"] == LGBM_PARAMS["min_child_samples"]