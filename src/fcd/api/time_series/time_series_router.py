from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from statsforecast.models import _TS  # type: ignore

from ...model.time_series import predict
from ..model_loader import load_models
from .time_series_models import PredictionRequest, PredictionResponse

router = APIRouter(
    prefix="/ts",
    tags=["time_series"],
)


@router.post("/predict", response_model=PredictionResponse)
async def ts_predict(
    request: PredictionRequest,
    models: Annotated[dict[str, _TS], Depends(load_models)],
) -> PredictionResponse:
    """
    Make a prediction using the trained model.

    Args:
        request: PredictionRequest containing h, level, and unique_id

    Returns:
        PredictionResponse with the prediction
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        prediction = predict(
            models, request.h, request.level, request.unique_id
        )

        return PredictionResponse(prediction=prediction)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
