from fastapi import FastAPI, HTTPException, status

from services.retraining_api.app.retraining import run_retraining
from services.retraining_api.app.schemas import RetrainingResponse

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Atlas CISIA - Retraining API",
              version="1.0.0",
              description="Service d'administration ML réservé aux équipes MLOps.")

Instrumentator(should_group_status_codes=False).instrument(app).expose(app,
                                                                       endpoint="/metrics",
                                                                       include_in_schema=False)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/retraining",
    response_model=RetrainingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def retraining() -> RetrainingResponse:
    try:
        result = run_retraining()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Échec du réentraînement : {type(error).__name__}") from error
    # Afficher le message pour debug
    # except Exception as error:
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail=f"Échec du réentraînement : {error}",
    #     ) from error

    return RetrainingResponse(**result)