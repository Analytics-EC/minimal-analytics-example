from pydantic import BaseModel


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction."""

    h: int
    level: list[int]
    unique_id: str


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    prediction: dict[str, list[float]]
