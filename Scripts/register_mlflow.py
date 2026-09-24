from __future__ import annotations

import json

import mlflow

from config import (
    CANDIDATE_VERSION,
    LGBM_PARAMS,
    MODEL_NAME,
    MODEL_SCENARIO,
    MODELS_DIR,
    REPORTS_DIR,
    MLFLOW_EXPERIMENT_NAME, 
    MLFLOW_TRACKING_URI
)

def main() -> None:
    """Journalise le candidat et son évaluation de référence dans MLflow."""
    report_path = REPORTS_DIR / "candidate_evaluation.json"
    model_path = MODELS_DIR / f"pipeline_{CANDIDATE_VERSION}.joblib"
    metadata_path = MODELS_DIR / f"pipeline_{CANDIDATE_VERSION}.json"

    if not report_path.exists():
        raise FileNotFoundError(
            "Rapport absent : exécutez d'abord evaluate_candidate.py."
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate_metrics = report["candidate"]["metrics"]

    print(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Rapport : {report_path}")
    print(f"Modèle : {model_path}")
    print(f"Métadonnées : {metadata_path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{MODEL_NAME}_{CANDIDATE_VERSION}"):
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "scenario": MODEL_SCENARIO,
                "model_version": CANDIDATE_VERSION,
                **LGBM_PARAMS,
            }
        )

        mlflow.log_metrics(
            {
                "reference_accuracy": candidate_metrics["accuracy"],
                "reference_f1_macro": candidate_metrics["f1_macro"],
                "reference_critical_errors_2_to_0": candidate_metrics[
                    "critical_errors_2_to_0"
                ],
                "reference_critical_error_rate_2_to_0": candidate_metrics[
                    "critical_error_rate_2_to_0"
                ]
                or 0.0,
            }
        )

        mlflow.log_artifact(report_path, artifact_path="reports")
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(metadata_path, artifact_path="model")

    print("Run MLflow enregistré.")


if __name__ == "__main__":
    main()