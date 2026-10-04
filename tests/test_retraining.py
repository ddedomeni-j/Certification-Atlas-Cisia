import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

from services.retraining_api.app import retraining


def test_build_enriched_training_set_adds_corrected_feedback(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "training_set.csv"
    source_path.write_text(
        (
            "age,niveau_diplome,anciennete_poste_ans,"
            "code_rome_vise,code_insee_commune,"
            "est_allocataire,nationalite_hors_ue,"
            "synthese_entretien,classe_retour_emploi\n"
            "30,Bac,2.0,M1805,75056,0,0,Recherche active.,1\n"
        ),
        encoding="utf-8",
    )

    enriched_path = tmp_path / "training_set_with_feedback.csv"

    monkeypatch.setattr(retraining, "TRAINING_DATA_PATH", source_path)
    monkeypatch.setattr(retraining, "RETRAINING_DATA_PATH", enriched_path)

    feedbacks = [
        {
            "feedback_id": 7,
            "input_data": {
                "age": 42,
                "niveau_diplome": "Bac+2",
                "anciennete_poste_ans": 4.0,
                "code_rome_vise": "M1805",
                "code_insee_commune": "75056",
                "est_allocataire": 1.0,
                "nationalite_hors_ue": 0,
                "synthese_entretien": "Formation terminée.",
            },
            "corrected_class": 2,
        }
    ]

    result_path = retraining.build_enriched_training_set(feedbacks)

    assert result_path == enriched_path

    content = enriched_path.read_text(encoding="utf-8")
    assert "Formation terminée." in content
    assert content.rstrip().endswith(",2")


def test_run_retraining_marks_feedbacks_only_after_success(
    monkeypatch,
    tmp_path,
):
    feedbacks = [
        {
            "feedback_id": 12,
            "input_data": {"age": 35},
            "corrected_class": 1,
        }
    ]
    executed_scripts = []
    marked_feedback_ids = []

    decision_path = tmp_path / "promotion_decision.json"
    decision_path.write_text(
        json.dumps({"reason": "Promotion recommandée.", "promote": True}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retraining,
        "list_feedbacks_for_training",
        lambda database_path: feedbacks,
    )
    monkeypatch.setattr(
        retraining,
        "build_enriched_training_set",
        lambda received_feedbacks: tmp_path / "training_set_with_feedback.csv",
    )
    monkeypatch.setattr(
        retraining,
        "run_script",
        lambda script_name, *arguments: executed_scripts.append(
            (script_name, arguments)
        ),
    )
    monkeypatch.setattr(retraining, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        retraining,
        "mark_feedbacks_as_used_for_training",
        lambda database_path, feedback_ids: marked_feedback_ids.extend(
            feedback_ids
        ),
    )

    result = retraining.run_retraining()

    assert result == {
        "status": "completed",
        "feedback_count": 1,
        "candidate_version": "candidate",
        "recommendation": "Promotion recommandée.",
    }
    assert executed_scripts == [
        (
            "train_and_persist.py",
            (
                "--data-path",
                str(tmp_path / "training_set_with_feedback.csv"),
            ),
        ),
        ("evaluate_candidate.py", ()),
        ("register_mlflow.py", ()),
        ("promotion.py", ()),
    ]
    assert marked_feedback_ids == [12]


def test_run_retraining_does_not_mark_feedbacks_when_training_fails(
    monkeypatch,
):
    feedbacks = [
        {
            "feedback_id": 12,
            "input_data": {"age": 35},
            "corrected_class": 1,
        }
    ]
    marked_feedback_ids = []

    monkeypatch.setattr(
        retraining,
        "list_feedbacks_for_training",
        lambda database_path: feedbacks,
    )
    monkeypatch.setattr(
        retraining,
        "build_enriched_training_set",
        lambda received_feedbacks: Path("unused.csv"),
    )

    def fail_training(script_name, *arguments):
        raise RuntimeError("train_and_persist.py a échoué")

    monkeypatch.setattr(retraining, "run_script", fail_training)
    monkeypatch.setattr(
        retraining,
        "mark_feedbacks_as_used_for_training",
        lambda database_path, feedback_ids: marked_feedback_ids.extend(
            feedback_ids
        ),
    )

    with pytest.raises(RuntimeError, match="a échoué"):
        retraining.run_retraining()

    assert marked_feedback_ids == []