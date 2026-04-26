"""Prediction functions for the diabetes model."""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict

from src.utils import get_feature_names, get_model_path


class DiabetesPredictor:
    """Diabetes prediction model wrapper."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        if model_path is None:
            model_path = get_model_path()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'python -m src.train' first."
            )
        
        self.model = joblib.load(model_path)
        self.feature_names = get_feature_names()
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Make a single prediction.
        
        Args:
            features: Dictionary with feature names and values
        
        Returns:
            Dictionary with prediction and probability
        """
        # Create DataFrame with correct feature order
        df = pd.DataFrame([features])[self.feature_names]
        
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return {
            'prediction': int(prediction),
            'label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability_non_diabetic': round(float(probability[0]), 4),
            'probability_diabetic': round(float(probability[1]), 4),
            'confidence': round(float(max(probability)), 4)
        }
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(features) for features in features_list]


# Singleton instance for the API
_predictor = None


def get_predictor() -> DiabetesPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DiabetesPredictor()
    return _predictor
