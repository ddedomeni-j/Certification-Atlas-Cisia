from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lightgbm import LGBMClassifier
from typing import Any

from evaluate import evaluate_model

from sklearn.model_selection import train_test_split
from data_preparation import build_preprocessor

from sklearn.pipeline import Pipeline

from config import LGBM_PARAMS, RANDOM_STATE

@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict[str, Any]
    best_params: dict[str, object]
    train_rows: int
    holdout_rows: int


def build_pipeline() -> Pipeline:
    """Construit le pipeline complet : preprocessing puis classifieur."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LGBMClassifier(**LGBM_PARAMS)),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> TrainingResult:
    """Entraîne le meilleur pipeline et l'évalue sur un holdout indépendant."""
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_holdout, y_holdout)

    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        best_params=LGBM_PARAMS,
        train_rows=len(X_train),
        holdout_rows=len(X_holdout),
    )