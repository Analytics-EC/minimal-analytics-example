# Regression API

A minimal FastAPI application for linear regression predictions, designed for deployment on Render.

## Features

- FastAPI-based REST API
- Linear regression model for predictions
- Automatic model training and loading
- Health check endpoint
- Ready for Render deployment

## Project Structure

```
minimal-analytics-example/
├── src/
│   └── regression_api/
│       ├── __init__.py
│       ├── main.py          # FastAPI application
│       └── model.py         # Model training and utilities
├── requirements.txt         # Python dependencies
├── setup.py                # Package configuration
├── render.yaml             # Render deployment configuration
└── README.md               # This file
```

## Local Development

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd minimal-analytics-example
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
# Development (with auto-reload)
uvicorn src.regression_api.main:app --reload

# Production (with gunicorn)
gunicorn src.regression_api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

The API will be available at `http://localhost:8000`

### Testing

The project includes comprehensive unit and integration tests using pytest.

#### Run All Tests
```bash
pytest
```

#### Run Specific Test Suites
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_model.py
```

#### Test Coverage
```bash
# Install coverage (if not already installed)
pip install pytest-cov

# Run tests with coverage
pytest --cov=src/regression_api --cov-report=html
```

### API Endpoints

- `GET /` - Home endpoint
- `GET /health` - Health check
- `POST /predict` - Make predictions

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [6]}'
```

## Deployment on Render

1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. Connect your repository to Render
3. Render will automatically detect the `render.yaml` configuration
4. The application will be deployed and available at the provided URL

## Model Details

The application uses a simple linear regression model trained on sample data:
- Features: [1, 2, 3, 4, 5]
- Targets: [20, 35, 41, 50, 78]

The model is automatically trained and saved on first startup if no existing model is found.
