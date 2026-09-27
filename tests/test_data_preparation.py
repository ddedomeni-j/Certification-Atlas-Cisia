import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))

from data_preparation import (
    build_preprocessor,
    extract_departement,
    prepare_features,
)


def test_extract_departement_handles_standard_and_overseas_codes():
    assert extract_departement("18273") == "18"
    assert extract_departement("2A004") == "2A"
    assert extract_departement("97100") == "971"
    assert extract_departement(None) == "INCONNU"


def test_prepare_features_creates_derived_columns_and_keeps_missing_text():
    dataframe = pd.DataFrame(
        {
            "code_insee_commune": ["18273"],
            "code_rome_vise": ["M1805"],
            "synthese_entretien": [None],
        }
    )

    result = prepare_features(dataframe)

    assert result.loc[0, "departement"] == "18"
    assert result.loc[0, "synthese_entretien"] == "TEXTE_MANQUANT"


def test_s5_preprocessor_contains_numeric_and_categorical_transformers():
    preprocessor = build_preprocessor()

    assert [name for name, _, _ in preprocessor.transformers] == ["num", "cat"]
    assert preprocessor.transformers[1][2][-1] == "synthese_entretien"