import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.backend_metier_api.app import model_client


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.request_data: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def post(self, url: str, json: dict, headers: dict):
        self.request_data = {
            "url": url,
            "json": json,
            "headers": headers,
        }
        return self.response


def test_predict_with_model_forwards_request_and_returns_response(monkeypatch):
    response_payload = {
        "prediction": 1,
        "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
        "model_version": "v1",
        "request_id": "request-123",
    }
    fake_client = FakeAsyncClient(FakeResponse(200, response_payload))

    monkeypatch.setattr(
        model_client.httpx,
        "AsyncClient",
        lambda **kwargs: fake_client,
    )

    result = asyncio.run(
        model_client.predict_with_model(
            application={"age": 35},
            request_id="request-123",
        )
    )

    assert result == response_payload
    assert fake_client.request_data is not None
    assert fake_client.request_data["json"] == {"age": 35}
    assert fake_client.request_data["headers"]["X-Request-ID"] == "request-123"


def test_predict_with_model_converts_model_api_error_to_502(monkeypatch):
    fake_client = FakeAsyncClient(FakeResponse(500, {"detail": "Erreur"}))

    monkeypatch.setattr(
        model_client.httpx,
        "AsyncClient",
        lambda **kwargs: fake_client,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            model_client.predict_with_model(
                application={"age": 35},
                request_id="request-123",
            )
        )

    assert error.value.status_code == 502