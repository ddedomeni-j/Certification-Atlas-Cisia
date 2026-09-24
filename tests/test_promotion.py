import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))

from promotion import decide_promotion


def test_rejects_candidate_below_f1_quality_floor():
    decision = decide_promotion(
        candidate={
            "f1_macro": 0.64,
            "critical_error_rate_2_to_0": 0.05,
        },
        production={
            "f1_macro": 0.70,
            "critical_error_rate_2_to_0": 0.10,
        },
    )

    assert decision.promote is False
    assert "sous le plancher" in decision.reason


def test_rejects_candidate_with_critical_error_regression():
    decision = decide_promotion(
        candidate={
            "f1_macro": 0.72,
            "critical_error_rate_2_to_0": 0.12,
        },
        production={
            "f1_macro": 0.70,
            "critical_error_rate_2_to_0": 0.10,
        },
    )

    assert decision.promote is False
    assert "erreurs critiques" in decision.reason


def test_rejects_candidate_without_minimum_f1_gain():
    decision = decide_promotion(
        candidate={
            "f1_macro": 0.702,
            "critical_error_rate_2_to_0": 0.10,
        },
        production={
            "f1_macro": 0.70,
            "critical_error_rate_2_to_0": 0.10,
        },
    )

    assert decision.promote is False
    assert "gain F1 macro insuffisant" in decision.reason


def test_promotes_candidate_with_f1_gain_and_no_critical_regression():
    decision = decide_promotion(
        candidate={
            "f1_macro": 0.71,
            "critical_error_rate_2_to_0": 0.10,
        },
        production={
            "f1_macro": 0.70,
            "critical_error_rate_2_to_0": 0.10,
        },
    )

    assert decision.promote is True
    assert "Promotion" in decision.reason