"""Utility functions for the diabetes prediction pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the diabetes dataset.
    - Replace zero values with NaN for certain columns
    - Fill NaN with median values
    """
    df = df.copy()
    
    # Columns where 0 is not a valid value
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_invalid_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    return df


def get_feature_names() -> list:
    """Return the list of feature names used in the model."""
    return [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_model_path() -> Path:
    """Get the path to the saved model."""
    return get_project_root() / 'models' / 'diabetes_model.pkl'


def get_data_path() -> Path:
    """Get the path to the dataset."""
    return get_project_root() / 'data' / 'diabetes.csv'
