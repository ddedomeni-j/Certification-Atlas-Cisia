from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """Évalue un pipeline entraîné sur un jeu de données étiqueté."""
    predictions = pipeline.predict(X)

    critical_errors = int(((y == 2) & (predictions == 0)).sum())
    class_2_count = int((y == 2).sum())

    return {
        "rows_evaluated": len(X),
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro")),
        "critical_errors_2_to_0": critical_errors,
        "critical_error_rate_2_to_0": (
            critical_errors / class_2_count if class_2_count else None
        ),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }