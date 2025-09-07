"""
Model training and loading utilities for linear regression.
"""

import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_model() -> LinearRegression:
    """
    Train a linear regression model with sample data.

    Returns:
        LinearRegression: Trained model
    """
    # Sample data
    x_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y_train = np.array([20, 35, 41, 50, 78])
    x_test = np.array([6, 7, 8, 9, 10]).reshape(-1, 1)
    y_test = np.array([80, 95, 101, 110, 128])

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Calculate predictions for evaluation
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")

    return model


def save_model(
        model: LinearRegression,
        filepath: str = "linear_regression.pkl") -> None:
    """
    Save the trained model to a pickle file.

    Args:
        model: Trained LinearRegression model
        filepath: Path where to save the model
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def load_model(
    filepath: str = "linear_regression.pkl",
) -> LinearRegression | None:
    """
    Load a trained model from a pickle file.

    Args:
        filepath: Path to the model file

    Returns:
        LinearRegression model or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        print(f"Error: Model file not found at {filepath}")

        return None

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {filepath}")

    return model


def predict(model: LinearRegression, features: list[float]) -> float:
    """
    Make a prediction using the trained model.

    Args:
        model: Trained LinearRegression model
        features: List of feature values

    Returns:
        Prediction value
    """
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)

    return float(prediction[0])
