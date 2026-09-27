import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

import promote_candidate as promotion


def write_json(path: Path, content: dict) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def configure_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    models_dir = tmp_path / "Models"
    reports_dir = tmp_path / "Reports"
    models_dir.mkdir()
    reports_dir.mkdir()

    active_model_path = models_dir / "active_model.json"

    monkeypatch.setattr(promotion, "MODELS_DIR", models_dir)
    monkeypatch.setattr(promotion, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(promotion, "ACTIVE_MODEL_PATH", active_model_path)

    return models_dir, reports_dir


def test_promote_candidate_versions_artifact_and_activates_it(monkeypatch, tmp_path):
    models_dir, reports_dir = configure_paths(monkeypatch, tmp_path)

    candidate_model_path = models_dir / "pipeline_candidate.joblib"
    candidate_model_path.write_bytes(b"candidate-model-content")

    write_json(
        models_dir / "pipeline_candidate.json",
        {
            "model_name": "S5_LightGBM",
            "model_version": "candidate",
            "artifact": "pipeline_candidate.joblib",
        },
    )

    write_json(
        reports_dir / "promotion_decision.json",
        {"promote": True, "reason": "Critères atteints."},
    )

    write_json(models_dir / "active_model.json", {"active_version": "v3"})

    promotion.promote_candidate("v4")

    promoted_model_path = models_dir / "pipeline_v4.joblib"
    promoted_metadata_path = models_dir / "pipeline_v4.json"

    assert promoted_model_path.exists()
    assert promoted_model_path.read_bytes() == b"candidate-model-content"

    metadata = json.loads(promoted_metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_version"] == "v4"
    assert metadata["artifact"] == "pipeline_v4.joblib"

    active_model = json.loads(
        (models_dir / "active_model.json").read_text(encoding="utf-8")
    )
    assert active_model == {"active_version": "v4"}


def test_promote_candidate_refuses_negative_recommendation(monkeypatch, tmp_path):
    models_dir, reports_dir = configure_paths(monkeypatch, tmp_path)

    write_json(
        reports_dir / "promotion_decision.json",
        {"promote": False, "reason": "F1 macro insuffisant."},
    )

    write_json(models_dir / "active_model.json", {"active_version": "v3"})

    with pytest.raises(RuntimeError, match="Promotion refusée"):
        promotion.promote_candidate("v4")

    assert not (models_dir / "pipeline_v4.joblib").exists()

    active_model = json.loads(
        (models_dir / "active_model.json").read_text(encoding="utf-8")
    )
    assert active_model == {"active_version": "v3"}