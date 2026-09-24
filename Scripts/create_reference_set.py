from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASET_PATH, RANDOM_STATE, TARGET_COLUMN

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "Data"
REFERENCE_PATH = REFERENCE_DIR / "reference_set.csv"
TRAINING_PATH = REFERENCE_DIR / "training_set.csv"

REFERENCE_SIZE = 0.20


def main() -> None:
    dataframe = pd.read_csv(
        DATASET_PATH,
        dtype={
            "code_insee_commune": "string",
            "code_rome_vise": "string",
        },
    )

    training_set, reference_set = train_test_split(
        dataframe,
        test_size=REFERENCE_SIZE,
        stratify=dataframe[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    if REFERENCE_PATH.exists() or TRAINING_PATH.exists():
        raise FileExistsError(
            "Jeu de référence ou jeu d'entraînement déjà présent. "
            "Ne les recréez pas sans une décision explicite de versionnement."
        )

    training_set.to_csv(TRAINING_PATH, index=False)
    reference_set.to_csv(REFERENCE_PATH, index=False)

    print(f"Jeu d'entraînement : {TRAINING_PATH} ({len(training_set)} lignes)")
    print(f"Jeu de référence : {REFERENCE_PATH} ({len(reference_set)} lignes)")


if __name__ == "__main__":
    main()