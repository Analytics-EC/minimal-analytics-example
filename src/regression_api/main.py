"""
FastAPI application for linear regression predictions.
"""
import os
import uvicorn

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import load_model, predict, train_model, save_model

# Initialize FastAPI app
app = FastAPI(
    title="Regression API",
    description="A minimal API for linear regression predictions",
    version="0.1.0"
)

# Global model variable
model = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    global model
    
    # Try to load existing model
    model_path = "modelo_regresion_lineal_up.pkl"
    model = load_model(model_path)
    
    # If model doesn't exist, train and save a new one
    if model is None:
        print("Training new model...")
        model = train_model()
        save_model(model, model_path)
        print("Model trained and saved successfully")
    
    print(f"Model loaded successfully: {model is not None}")


@app.get("/", response_model=dict)
async def home():
    """Home endpoint."""
    return {"message": "API for Linear Regression Model"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """
    Make a prediction using the trained model.
    
    Args:
        request: PredictionRequest containing features
        
    Returns:
        PredictionResponse with the prediction
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        prediction = predict(model, request.features)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
