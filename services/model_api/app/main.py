from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import MODEL_VERSION, MODELS_DIR
from services.model_api.app.middleware import LoggingMiddleware
from services.model_api.app.model_loader import load_model_and_metadata
from services.model_api.app.schemas import (
    EmploymentApplication,
    HealthResponse,
    InfoResponse,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le pipeline S5 actif une fois au démarrage de l'API."""
    model, metadata = load_model_and_metadata(MODELS_DIR, MODEL_VERSION)

    app.state.model = model
    app.state.metadata = metadata

    logger.info(
        "Modèle chargé : {model_name} {model_version}",
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
    )

    yield

    app.state.model = None
    app.state.metadata = None
    logger.info("Modèle libéré")


app = FastAPI(
    title="Atlas CISIA - Model API",
    version="1.0.0",
    description="Service interne de prédiction du modèle S5 LightGBM.",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Indique si le modèle est bien chargé."""
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé.",
        )

    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    """Retourne les métadonnées de l'artefact actif."""
    metadata = app.state.metadata

    return InfoResponse(
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        scenario=metadata["scenario"],
        model_created_at=metadata["created_at"],
        feature_columns=metadata["feature_columns"],
        metrics_holdout=metadata["metrics_holdout"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    application: EmploymentApplication,
    request: Request,
) -> PredictionResponse:
    """Prédit la classe de retour à l'emploi et ses probabilités."""
    request_id = request.state.request_id
    model = app.state.model
    metadata = app.state.metadata

    try:
        features = pd.DataFrame([application.model_dump()])
        prediction = int(model.predict(features)[0])
        raw_probabilities = model.predict_proba(features)[0]

        probabilities = {
            int(class_label): round(float(probability), 4)
            for class_label, probability in zip(model.classes_, raw_probabilities)
        }
    except Exception as error:
        logger.bind(request_id=request_id).exception("Échec de prédiction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Échec de prédiction : {type(error).__name__}",
        ) from error

    return PredictionResponse(
        prediction=prediction,
        probabilities=probabilities,
        model_version=metadata["model_version"],
        request_id=request_id,
    )