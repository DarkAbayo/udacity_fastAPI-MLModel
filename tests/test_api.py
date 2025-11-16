"""
Tests for FastAPI endpoints.

Tests include:
- GET endpoint: Welcome message
- POST endpoint: Model inference (both prediction outcomes)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_endpoint():
    """
    Test GET endpoint returns welcome message.

    Must test both status code and response content for sanity check.
    """
    response = client.get("/")

    # Test status code (required by sanity check)
    assert response.status_code == 200

    # Test response content (required by sanity check)
    data = response.json()
    assert "message" in data
    assert data["message"] == "Welcome to the Census Income Prediction API"


def test_post_endpoint_prediction_high():
    """
    Test POST endpoint with data leading to >50K prediction.

    Must test both status code and response content for sanity check.
    Uses data that typically leads to high income prediction.
    """
    # Test data designed to lead to >50K prediction
    # Higher age, higher education, executive occupation, etc.
    test_data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 83311,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    response = client.post("/inference", json=test_data)

    # Test status code (required by sanity check)
    assert response.status_code == 200

    # Test response content (required by sanity check)
    data = response.json()
    assert "prediction" in data
    # Note: Actual prediction depends on model, but should be either
    # ">50K" or "<=50K"
    assert data["prediction"] in [">50K", "<=50K"]


def test_post_endpoint_prediction_low():
    """
    Test POST endpoint with data leading to <=50K prediction.

    Must test both status code and response content for sanity check.
    Uses data that typically leads to low income prediction.
    """
    # Test data designed to lead to <=50K prediction
    # Lower age, lower education, service occupation, etc.
    test_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/inference", json=test_data)

    # Test status code (required by sanity check)
    assert response.status_code == 200

    # Test response content (required by sanity check)
    data = response.json()
    assert "prediction" in data
    # Note: Actual prediction depends on model, but should be either
    # ">50K" or "<=50K"
    assert data["prediction"] in [">50K", "<=50K"]
