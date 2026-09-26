from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import HTTPException, status

MODEL_API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8000")

async def predict_with_model(
    application: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Appelle le service interne de prédiction."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MODEL_API_URL}/predict",
                json=application,
                headers={"X-Request-ID": request_id},
            )
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service de prédiction indisponible.",
        ) from error

    if response.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Le service de prédiction a retourné une erreur.",
        )

    return response.json()