from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import numpy as np

from config import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def extract_departement(code_insee: object) -> str:
    if pd.isna(code_insee):
        return "INCONNU"

    code = str(code_insee).strip().upper().removesuffix(".0")

    if code.isdigit() and len(code) < 5:
        code = code.zfill(5)

    if code.startswith(("971", "972", "973", "974", "975", "976")):
        return code[:3]

    if code.startswith(("2A", "2B")):
        return code[:2]

    return code[:2] if len(code) >= 2 else "INCONNU"


def prepare_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["departement"] = dataframe["code_insee_commune"].map(
        extract_departement
    )

    dataframe["synthese_entretien"] = (dataframe["synthese_entretien"]
                                            .fillna("")
                                            .astype(str)
                                            .str.strip()
                                            .replace("", "TEXTE_MANQUANT"))

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    for column in feature_columns:
        if column not in dataframe.columns:
            continue

        dataframe[column] = dataframe[column].astype(object)
        dataframe.loc[dataframe[column].isna(), column] = np.nan

    return dataframe


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    dataframe = pd.read_csv(
        path,
        dtype={
            "code_insee_commune": "string",
            "code_rome_vise": "string",
        },
    )

    required_columns = {
        "age",
        "niveau_diplome",
        "anciennete_poste_ans",
        "code_rome_vise",
        "code_insee_commune",
        "est_allocataire",
        "nationalite_hors_ue",
        "synthese_entretien",
        TARGET_COLUMN,
    }
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise KeyError(f"Colonnes absentes : {missing_columns}")

    dataframe = prepare_features(dataframe)
    X = dataframe[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = dataframe[TARGET_COLUMN].copy()

    if y.isna().any():
        raise ValueError("La variable cible contient des valeurs manquantes.")

    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )