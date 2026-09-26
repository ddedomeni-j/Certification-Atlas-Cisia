from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    BACKEND_DB_PATH,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

from data_preparation import prepare_features

from services.backend_metier_api.app.model_client import predict_with_model
from services.backend_metier_api.app.repository import (
    get_prediction,
    init_database,
    list_predictions,
    save_feedback,
    save_prediction,
)
from services.backend_metier_api.app.schemas import (
    EmploymentApplication,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictionResult,
    PredictionSummary,
)

from prometheus_fastapi_instrumentator import Instrumentator

from services.backend_metier_api.app.metrics import observe_feedback

import pandas as pd

DECISION_LABELS = {
    0: "Retour rapide à l'emploi",
    1: "Retour à l'emploi intermédiaire",
    2: "Risque de chômage de longue durée",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise la base SQLite au démarrage du backend métier."""
    init_database(BACKEND_DB_PATH)
    yield

app = FastAPI(title="Atlas CISIA - API métier",
              version="1.0.0",
              description="API métier entre le frontend et le service S5 LightGBM.",
              lifespan=lifespan)

Instrumentator(should_group_status_codes=False).instrument(app).expose(app,
                                                                       endpoint="/metrics",
                                                                       include_in_schema=False)

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/applications/predict",
          response_model=PredictionResult,
          status_code=status.HTTP_200_OK,
)
async def predict_application(
    application: EmploymentApplication,
) -> PredictionResult:
    """Score un dossier et journalise la prédiction."""
    request_id = str(uuid4())

    raw_data = application.model_dump()

    prepared = prepare_features(pd.DataFrame([raw_data]))

    model_input = prepared[NUMERIC_FEATURES + CATEGORICAL_FEATURES].iloc[0].to_dict()

    model_response = await predict_with_model(application=model_input,
                                              request_id=request_id)

    prediction = int(model_response["prediction"])
    if prediction not in DECISION_LABELS:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Classe invalide retournée par le service de prédiction.")

    created_at = datetime.now(timezone.utc)
    decision = DECISION_LABELS[prediction]

    save_prediction(BACKEND_DB_PATH,
                    request_id=model_response["request_id"],
                    input_data=raw_data,
                    prediction=prediction,
                    probabilities=model_response["probabilities"],
                    decision=decision,
                    model_version=model_response["model_version"],
                    created_at=created_at)

    return PredictionResult(request_id=model_response["request_id"],
                            prediction=prediction,
                            decision=decision,
                            probabilities=model_response["probabilities"],
                            model_version=model_response["model_version"],
                            created_at=created_at)


@app.get("/applications",
         response_model=list[PredictionSummary])
async def get_applications() -> list[PredictionSummary]:
    """Liste les prédictions journalisées pour la sélection d'un feedback."""
    predictions = list_predictions(BACKEND_DB_PATH)

    return [PredictionSummary(request_id=prediction["request_id"],
                              created_at=prediction["created_at"],
                              prediction=prediction["prediction"],
                              decision=prediction["decision"],
                              model_version=prediction["model_version"])
            for prediction in predictions]

@app.get("/applications/{request_id}",
         response_model=PredictionResult)
async def get_application_prediction(request_id: str) -> PredictionResult:
    """Retourne une prédiction précédemment journalisée."""
    prediction = get_prediction(BACKEND_DB_PATH, request_id)

    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Prédiction introuvable.")

    return PredictionResult(request_id=prediction["request_id"],
                            prediction=prediction["prediction"],
                            decision=prediction["decision"],
                            probabilities=prediction["probabilities"],
                            model_version=prediction["model_version"],
                            created_at=prediction["created_at"])


@app.post("/feedback",
          response_model=FeedbackResponse,
          status_code=status.HTTP_201_CREATED)
async def create_feedback(feedback: FeedbackRequest) -> FeedbackResponse:
    """Enregistre une correction métier liée à une prédiction."""
    try:
        feedback_id, created_at = save_feedback(BACKEND_DB_PATH,
                                                request_id=feedback.request_id,
                                                corrected_class=feedback.corrected_class,
                                                comment=feedback.comment)
        observe_feedback(feedback.corrected_class)
        
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Prédiction introuvable.") from error

    return FeedbackResponse(feedback_id=feedback_id,
                            request_id=feedback.request_id,
                            created_at=created_at)