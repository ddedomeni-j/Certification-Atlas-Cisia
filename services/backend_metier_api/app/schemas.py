from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

class EmploymentApplication(BaseModel):
    """Dossier transmis par le frontend pour obtenir une prédiction S5."""

    age: int = Field(..., ge=16, le=100)
    anciennete_poste_ans: float = Field(..., ge=0, le=60)
    niveau_diplome: str = Field(..., min_length=1)
    est_allocataire: float = Field(..., ge=0, le=1)
    nationalite_hors_ue: int = Field(..., ge=0, le=1)
    code_insee_commune: str = Field(..., min_length=5, max_length=5)
    code_rome_vise: str = Field(..., min_length=5, max_length=5)
    synthese_entretien: str = Field(..., min_length=1)


class PredictionResult(BaseModel):
    """Résultat métier retourné au frontend après scoring."""

    request_id: str
    prediction: int = Field(..., ge=0, le=2)
    decision: str
    review_status: str
    probabilities: dict[int, float]
    model_version: str
    created_at: datetime


class FeedbackRequest(BaseModel):
    """Correction apportée par un conseiller sur une prédiction."""

    request_id: str = Field(..., min_length=1)
    corrected_class: int = Field(..., ge=0, le=2)
    comment: str | None = Field(default=None, max_length=1_000)


class FeedbackResponse(BaseModel):
    """Confirmation d'enregistrement du feedback."""

    feedback_id: int
    request_id: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: str

class PredictionSummary(BaseModel):
    request_id: str
    created_at: datetime
    prediction: int = Field(..., ge=0, le=2)
    decision: str
    review_status: str
    model_version: str