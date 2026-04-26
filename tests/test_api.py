"""Tests for the diabetes prediction API."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self):
        """Test that health endpoint returns valid response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""
    
    def test_model_info(self):
        """Test that model info endpoint returns expected data."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "RandomForestClassifier"
        assert len(data["features"]) == 8
        assert "Glucose" in data["features"]


class TestPredictionEndpoint:
    """Tests for the prediction endpoint."""
    
    @pytest.fixture
    def valid_input(self):
        """Valid input data for testing."""
        return {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
    
    @pytest.fixture
    def low_risk_input(self):
        """Low risk patient data."""
        return {
            "Pregnancies": 1,
            "Glucose": 85,
            "BloodPressure": 66,
            "SkinThickness": 29,
            "Insulin": 0,
            "BMI": 26.6,
            "DiabetesPedigreeFunction": 0.351,
            "Age": 31
        }
    
    def test_valid_prediction(self, valid_input):
        """Test prediction with valid input."""
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "probability_diabetic" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]
        assert data["label"] in ["Diabetic", "Non-Diabetic"]
    
    def test_prediction_probabilities(self, valid_input):
        """Test that probabilities sum to 1."""
        response = client.post("/predict", json=valid_input)
        data = response.json()
        total_prob = data["probability_diabetic"] + data["probability_non_diabetic"]
        assert abs(total_prob - 1.0) < 0.01
    
    def test_invalid_input_negative_age(self, valid_input):
        """Test validation rejects negative age."""
        valid_input["Age"] = -5
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422
    
    def test_invalid_input_missing_field(self):
        """Test validation rejects missing fields."""
        incomplete_input = {"Pregnancies": 1, "Glucose": 100}
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422


class TestBatchPrediction:
    """Tests for batch prediction endpoint."""
    
    def test_batch_prediction(self):
        """Test batch prediction with multiple inputs."""
        batch_input = {
            "inputs": [
                {
                    "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
                    "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627, "Age": 50
                },
                {
                    "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66,
                    "SkinThickness": 29, "Insulin": 0, "BMI": 26.6,
                    "DiabetesPedigreeFunction": 0.351, "Age": 31
                }
            ]
        }
        response = client.post("/predict/batch", json=batch_input)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2


class TestMetricsEndpoint:
    """Tests for the metrics endpoint."""
    
    def test_metrics(self):
        """Test that metrics endpoint returns usage data."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "uptime_seconds" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
