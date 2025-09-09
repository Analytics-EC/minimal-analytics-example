from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    models_loaded: bool
