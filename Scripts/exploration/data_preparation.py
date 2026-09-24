# from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.lang.fr.stop_words import STOP_WORDS

# UTILS
# =====
def extract_departement(code_insee: object) -> str:
    """Extrait le département à partir d'un code INSEE commune.

    Gère notamment :
    - les départements métropolitains ;
    - la Corse avec 2A et 2B ;
    - les départements et régions d'outre-mer.
    """
    if pd.isna(code_insee):
        return "INCONNU"

    code = str(code_insee).strip().upper()

    # Nettoyage du cas où pandas a lu le code comme un float : 63000.0
    if code.endswith(".0"):
        code = code[:-2]

    # On remet les codes métropolitains sur cinq caractères
    if code.isdigit() and len(code) < 5:
        code = code.zfill(5)

    if code.startswith(("971", "972", "973", "974", "975", "976")):
        return code[:3]

    if code.startswith(("2A", "2B")):
        return code[:2]

    if len(code) >= 2:
        return code[:2]

    return "INCONNU"

def extract_domaine_rome(code_rome: object) -> str:
    """Extrait le domaine principal du code ROME.

    Exemple :
        M1805 -> M
        N4103 -> N
    """
    if pd.isna(code_rome):
        return "INCONNU"

    code = str(code_rome).strip().upper()

    if not code:
        return "INCONNU"

    return code[0]

# Définition variables
TARGET = "classe_retour_emploi"
num_scenario = 3
if num_scenario == 1:
    NUMERIC_FEATURES = ["age",
                        "anciennete_poste_ans"]

    CATEGORICAL_FEATURES = ["niveau_diplome",
                            "est_allocataire",
                            "nationalite_hors_ue",
                            "departement",
                            "domaine_rome"]

    TEXT_FEATURE = "synthese_entretien"

    TARGET_COLUMN = "classe_retour_emploi"

    EXCLUDED_FEATURES = ["usager_id",
                         "code_insee_commune",
                         "code_rome_vise"]
elif num_scenario == 2:
    # On supprime la variable nationalite_hors_ue
    NUMERIC_FEATURES = ["anciennete_poste_ans"]

    CATEGORICAL_FEATURES = ["niveau_diplome",
                            "est_allocataire",
                            "departement",
                            "domaine_rome"]

    TEXT_FEATURE = "synthese_entretien"

    TARGET_COLUMN = "classe_retour_emploi"

    EXCLUDED_FEATURES = ["usager_id",
                         "age"
                         "nationalite_hors_ue",
                         "code_insee_commune",
                         "code_rome_vise"]
elif num_scenario == 3:
    NUMERIC_FEATURES = []
    
    CATEGORICAL_FEATURES = []

    TEXT_FEATURE = "synthese_entretien"

    TARGET_COLUMN = "classe_retour_emploi"

    EXCLUDED_FEATURES = ["usager_id",
                         "code_insee_commune",
                         "code_rome_vise",
                         "age",
                         "anciennete_poste_ans",
                         "niveau_diplome",
                         "est_allocataire",
                         "nationalite_hors_ue",
                         "departement",
                         "domaine_rome"]
else:
    # Seulement données tabulaires
    NUMERIC_FEATURES = ["age",
                        "anciennete_poste_ans"]

    CATEGORICAL_FEATURES = ["niveau_diplome",
                            "code_rome_vise"]

    # Variables explicitement exclues
    EXCLUDED_FEATURES = ["usager_id",
                        "code_insee_commune",
                        "nationalite_hors_ue"]

# def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
#     df = pd.read_csv(path)
#     y = df[TARGET]
#     features = (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
#     X = df[features].copy()
#     return X, y

def load_dataset_scenario1(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Charge les données, crée les caractéristiques et retourne X et y."""
    path = Path(path)

    df = pd.read_csv(path,
                     sep=",",
                     dtype={"code_insee_commune": "string",
                            "code_rome_vise": "string"})

    required_columns = ["usager_id",
                        "age",
                        "niveau_diplome",
                        "anciennete_poste_ans",
                        "code_rome_vise",
                        "code_insee_commune",
                        "est_allocataire",
                        "nationalite_hors_ue",
                        "synthese_entretien",
                        TARGET_COLUMN]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Colonnes absentes du fichier : {missing_columns}")

    # Feature engineering
    df["departement"] = (df["code_insee_commune"].apply(extract_departement))

    df["domaine_rome"] = (df["code_rome_vise"].apply(extract_domaine_rome))

    # Pour le texte : faire apparaitre texte manquant pour faire apparaitre le risque
    # df[TEXT_FEATURE] = (df[TEXT_FEATURE].fillna("TEXTE_MANQUANT").replace(r"^\s*$", "TEXTE_MANQUANT", regex=True).astype(str))
    mask = (df["synthese_entretien"].fillna("").str.strip() != "")
    df = df[mask]
    
    # # Le remplacement par une chaîne vide ne nécessite aucun apprentissage.
    # # Il n'entraîne donc pas de fuite de données.
    # df[TEXT_FEATURE] = (df[TEXT_FEATURE].fillna("").astype(str))

    selected_features = (NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE])
    
    X = df[selected_features].copy()
    y = df[TARGET_COLUMN].copy()

    if y.isna().any():
        raise ValueError("La variable cible contient des valeurs manquantes.")

    return X, y

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer(transformers=[("num", numeric_pipeline, NUMERIC_FEATURES),
                                           ("cat", categorical_pipeline, CATEGORICAL_FEATURES)],
                             sparse_threshold=0)

def build_preprocessor_scenario1() -> ColumnTransformer:
    """Construit le préprocesseur multimodal du scénario 1."""

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                           ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    text_pipeline = TfidfVectorizer(lowercase=True,
                                    strip_accents="unicode",
                                    max_features=2000,
                                    min_df=3,
                                    max_df=0.95,
                                    ngram_range=(1, 2),
                                    stop_words=list(STOP_WORDS),
                                    sublinear_tf=True)

    return ColumnTransformer(transformers=[("num", numeric_pipeline, NUMERIC_FEATURES),
                                           ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
                                           ("text", text_pipeline, TEXT_FEATURE)],
                             remainder="drop")

def build_preprocessor_scenario3():

    text_pipeline = TfidfVectorizer(lowercase=True,
                                    strip_accents="unicode",
                                    max_features=2000,
                                    min_df=3,
                                    max_df=0.95,
                                    ngram_range=(1, 2),
                                    stop_words=list(STOP_WORDS),
                                    sublinear_tf=True)

    return ColumnTransformer(transformers=[("text", text_pipeline, TEXT_FEATURE)], remainder="drop")