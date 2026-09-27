from __future__ import annotations

from pydantic import BaseModel, Field


class EmploymentApplication(BaseModel):
    """Données brutes nécessaires au pipeline S5 LightGBM."""

    age: float = Field(..., ge=16, le=100)
    anciennete_poste_ans: float = Field(..., ge=0, le=60)
    niveau_diplome: str = Field(..., min_length=1)
    est_allocataire: float = Field(..., ge=0, le=1)
    nationalite_hors_ue: int = Field(..., ge=0, le=1)
    departement: str = Field(..., min_length=2, max_length=3)
    code_rome_vise: str = Field(..., min_length=5, max_length=5)
    synthese_entretien: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=2)
    probabilities: dict[int, float]
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    model_name: str
    model_version: str
    scenario: str
    model_created_at: str
    feature_columns: list[str]
    metrics_holdout: dict