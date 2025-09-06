"""
FastAPI application for linear regression predictions.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import load_model, predict, save_model, train_model

# Global model storage
model = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator:
    """Manage the application lifespan events."""
    # Startup: Load the model when the application starts
    model_path = "modelo_regresion_lineal_up.pkl"
    model["pickle"] = load_model(model_path)

    # If model doesn't exist, train and save a new one
    if not model.get("pickle"):
        print("Training new model...")
        model["pickle"] = train_model()
        save_model(model, model_path)
        print("Model trained and saved successfully")

    print(f"Model loaded successfully: {model.get('pickle')}")

    yield

    # Shutdown: Clean up resources (if needed)
    print("Application shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Regression API",
    description="A minimal API for linear regression predictions",
    version="0.1.0",
    lifespan=lifespan,
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/", response_model=dict)
async def home() -> dict[str, str]:
    """Home endpoint."""
    return {"message": "API for Linear Regression Model"}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_loaded=model.get("pickle") is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """
    Make a prediction using the trained model.

    Args:
        request: PredictionRequest containing features

    Returns:
        PredictionResponse with the prediction
    """
    if not model.get("pickle"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        prediction = predict(model.get("pickle"), request.features)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
