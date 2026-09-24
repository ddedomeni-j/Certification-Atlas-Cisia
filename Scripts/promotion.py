from __future__ import annotations

from dataclasses import dataclass


import json

from config import REPORTS_DIR, MIN_F1_MACRO, F1_MIN_GAIN, CRITICAL_ERROR_RATE_TOLERANCE

@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str

def decide_promotion(
    candidate: dict[str, float | int | None],
    production: dict[str, float | int | None],
) -> PromotionDecision:
    """Décide la promotion selon une politique métier explicite."""
    candidate_f1 = float(candidate["f1_macro"])
    production_f1 = float(production["f1_macro"])

    candidate_critical_rate = float(candidate["critical_error_rate_2_to_0"] or 0)
    production_critical_rate = float(production["critical_error_rate_2_to_0"] or 0)

    if candidate_f1 < MIN_F1_MACRO:
        return PromotionDecision(
            promote=False,
            reason=(
                f"Rejet : F1 macro candidat {candidate_f1:.3f} "
                f"sous le plancher {MIN_F1_MACRO:.3f}."
            ),
        )

    if candidate_critical_rate > (
        production_critical_rate + CRITICAL_ERROR_RATE_TOLERANCE
    ):
        return PromotionDecision(
            promote=False,
            reason=(
                "Rejet : le taux d'erreurs critiques 2 -> 0 augmente de "
                f"{production_critical_rate:.3f} à {candidate_critical_rate:.3f}."
            ),
        )

    f1_gain = candidate_f1 - production_f1
    if f1_gain < F1_MIN_GAIN:
        return PromotionDecision(
            promote=False,
            reason=(
                f"Rejet : gain F1 macro insuffisant ({f1_gain:+.3f}), "
                f"minimum attendu {F1_MIN_GAIN:+.3f}."
            ),
        )

    return PromotionDecision(
        promote=True,
        reason=(
            f"Promotion : F1 macro {production_f1:.3f} -> {candidate_f1:.3f} "
            f"({f1_gain:+.3f}) et erreurs critiques non dégradées."
        ),
    )

def main() -> None:
    report_path = REPORTS_DIR / "candidate_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    decision = decide_promotion(
        candidate=report["candidate"]["metrics"],
        production=report["production"]["metrics"],
    )

    decision_path = REPORTS_DIR / "promotion_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "promote": decision.promote,
                "reason": decision.reason,
                "candidate": report["candidate"],
                "production": report["production"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(decision.reason)
    print(f"Décision enregistrée : {decision_path}")


if __name__ == "__main__":
    main()