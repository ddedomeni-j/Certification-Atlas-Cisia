from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
MODELS_DIR = PROJECT_ROOT / "Models"
TRAINING_DATA_PATH = DATA_DIR / "training_set.csv"
REFERENCE_SET_PATH = DATA_DIR / "reference_set.csv"

DATASET_NAME = (
    "dataset_trajectoire_emploi_Sujet Examen CISIA - "
    "Promo Upskilling Atlas - mai-oct2026 (Session-00279143).csv"
)
DATASET_PATH = DATA_DIR / DATASET_NAME

TARGET_COLUMN = "classe_retour_emploi"

NUMERIC_FEATURES = [
    "age",
    "anciennete_poste_ans",
]

CATEGORICAL_FEATURES = [
    "niveau_diplome",
    "est_allocataire",
    "nationalite_hors_ue",
    "departement",
    "domaine_rome",
    "synthese_entretien",
]

MODEL_NAME = "S5_LightGBM"
MODEL_VERSION = "v1"
MODEL_SCENARIO = "S5"
RANDOM_STATE = 42

CANDIDATE_VERSION = "candidate"
REPORTS_DIR = PROJECT_ROOT / "Reports"

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "num_leaves": 31,
    "min_child_samples": 20,
}

MIN_F1_MACRO = 0.65
F1_MIN_GAIN = 0.005
CRITICAL_ERROR_RATE_TOLERANCE = 0.01

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "atlas-cisia-s5"

BACKEND_DB_PATH = DATA_DIR / "backend_metier.db"