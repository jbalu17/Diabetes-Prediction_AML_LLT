"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Literal


class DiabetesInput(BaseModel):
    """Input schema for diabetes prediction."""
    
    Pregnancies: int = Field(
        ge=0, le=20,
        description="Number of pregnancies",
        examples=[6]
    )
    Glucose: float = Field(
        ge=0, le=300,
        description="Plasma glucose concentration (mg/dL)",
        examples=[148.0]
    )
    BloodPressure: float = Field(
        ge=0, le=200,
        description="Diastolic blood pressure (mm Hg)",
        examples=[72.0]
    )
    SkinThickness: float = Field(
        ge=0, le=100,
        description="Triceps skin fold thickness (mm)",
        examples=[35.0]
    )
    Insulin: float = Field(
        ge=0, le=900,
        description="2-Hour serum insulin (μU/mL)",
        examples=[0.0]
    )
    BMI: float = Field(
        ge=0, le=70,
        description="Body mass index (kg/m²)",
        examples=[33.6]
    )
    DiabetesPedigreeFunction: float = Field(
        ge=0, le=3,
        description="Diabetes pedigree function",
        examples=[0.627]
    )
    Age: int = Field(
        ge=1, le=120,
        description="Age in years",
        examples=[50]
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""
    
    prediction: int = Field(description="Binary prediction (0 or 1)")
    label: Literal["Diabetic", "Non-Diabetic"] = Field(
        description="Human-readable prediction label"
    )
    probability_non_diabetic: float = Field(
        ge=0, le=1,
        description="Probability of being non-diabetic"
    )
    probability_diabetic: float = Field(
        ge=0, le=1,
        description="Probability of being diabetic"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="Confidence of the prediction"
    )


class BatchInput(BaseModel):
    """Input schema for batch predictions."""
    
    inputs: List[DiabetesInput] = Field(
        description="List of patient data for batch prediction"
    )


class BatchResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(
        description="List of predictions"
    )
    count: int = Field(description="Number of predictions made")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    model_type: str = Field(description="Type of ML model")
    features: List[str] = Field(description="List of input features")
    version: str = Field(description="Model version")
