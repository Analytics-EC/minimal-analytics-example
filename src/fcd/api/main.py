"""
FastAPI application for linear regression predictions.
"""

import os
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI
from statsforecast.models import _TS  # type: ignore

from .health.health_models import HealthResponse
from .model_loader import load_models
from .time_series.time_series_router import router as ts_router

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Regression API",
    description="A minimal API for linear regression predictions",
    version="0.1.0",
)


@app.get("/", response_model=dict[str, str])
async def home() -> dict[str, str]:
    """Home endpoint."""
    return {"message": "API for Linear Regression Model"}


@app.get("/health", response_model=HealthResponse)
async def health_check(
    models: Annotated[dict[str, _TS], Depends(load_models)],
) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", models_loaded=models is not None)


# Register routers
app.include_router(
    ts_router, dependencies=[Depends(load_models, use_cache=False)]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
