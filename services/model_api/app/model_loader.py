from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def load_model_and_metadata(
    models_dir: Path,
    version: str,
) -> tuple[Any, dict[str, Any]]:
    """Charge un pipeline joblib et son fichier de métadonnées JSON."""
    model_path = models_dir / f"pipeline_{version}.joblib"
    metadata_path = models_dir / f"pipeline_{version}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Artefact modèle absent : {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Métadonnées du modèle absentes : {metadata_path}"
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return model, metadata