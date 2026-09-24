from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from config import MODEL_NAME, MODEL_SCENARIO, TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERIC_FEATURES

def save_model(
    pipeline: Pipeline,
    output_dir: str | Path,
    version: str,
    metrics: dict[str, Any],
    best_params: dict[str, object],
    train_rows: int,
    holdout_rows: int,
) -> tuple[Path, Path]:
    """Sauvegarde le pipeline complet et les métadonnées associées."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"pipeline_{version}.joblib"
    metadata_path = output_dir / f"pipeline_{version}.json"

    joblib.dump(pipeline, model_path, compress=3)

    metadata: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "scenario": MODEL_SCENARIO,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET_COLUMN,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "metrics_holdout": metrics,
        "best_params": best_params,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "artifact": model_path.name,
        "sklearn_version": sklearn.__version__,
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return model_path, metadata_path