from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "Models"
DATA_DIR = PROJECT_ROOT / "Data"
TRAINING_DATA_PATH = DATA_DIR / "training_set.csv"
REFERENCE_SET_PATH = DATA_DIR / "reference_set.csv"

ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.json"

def load_active_model_version() -> str:
    data = json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    version = data.get("active_version")

    if not isinstance(version, str) or not version:
        raise ValueError("active_model.json doit contenir active_version.")

    return version

MODEL_VERSION = load_active_model_version()

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
    "code_rome_vise",
    "synthese_entretien",
]

MODEL_NAME = "S5_LightGBM"
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

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "atlas-cisia-s5"

BACKEND_DB_PATH = DATA_DIR / "backend_metier.db"