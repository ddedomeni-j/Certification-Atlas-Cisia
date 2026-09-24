from __future__ import annotations

import argparse

from config import CANDIDATE_VERSION, MODELS_DIR, TRAINING_DATA_PATH
from data_preparation import load_dataset
from persist import save_model
from train import train_model


def main() -> None:
    """Entraîne S5 LightGBM et persiste un artefact candidat ou versionné."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        default=CANDIDATE_VERSION,
        help="Version de l'artefact à créer.",
    )
    arguments = parser.parse_args()

    X, y = load_dataset(TRAINING_DATA_PATH)
    result = train_model(X, y)

    model_path, metadata_path = save_model(
        pipeline=result.pipeline,
        output_dir=MODELS_DIR,
        version=arguments.version,
        metrics=result.metrics,
        best_params=result.best_params,
        train_rows=result.train_rows,
        holdout_rows=result.holdout_rows,
    )

    print(f"Modèle sauvegardé : {model_path}")
    print(f"Métadonnées sauvegardées : {metadata_path}")
    print(f"F1 macro holdout : {result.metrics['f1_macro']:.3f}")
    print(
        "Erreurs critiques 2 -> 0 : "
        f"{result.metrics['critical_errors_2_to_0']}"
    )


if __name__ == "__main__":
    main()