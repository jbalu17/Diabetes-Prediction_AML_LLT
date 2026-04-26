"""Training script for the diabetes prediction model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix
)
import joblib
from pathlib import Path

from src.utils import (
    load_data, preprocess_data, get_feature_names,
    get_model_path, get_data_path
)


def train_model(data_path: str = None, model_path: str = None):
    """
    Train the diabetes prediction model.
    
    Args:
        data_path: Path to the dataset CSV
        model_path: Path to save the trained model
    
    Returns:
        dict: Training metrics and model information
    """
    # Set default paths
    if data_path is None:
        data_path = get_data_path()
    if model_path is None:
        model_path = get_model_path()
    
    # Ensure model directory exists
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DIABETES PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1/5] Loading dataset...")
    df = load_data(data_path)
    print(f"   Dataset shape: {df.shape}")
    
    print("\n[2/5] Preprocessing data...")
    df = preprocess_data(df)
    
    # Split features and target
    feature_names = get_feature_names()
    X = df[feature_names]
    y = df['Outcome']
    
    print(f"   Features: {feature_names}")
    print(f"   Target distribution: {dict(y.value_counts())}")
    
    # Train-test split
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train model
    print("\n[4/5] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"   Cross-validation F1 scores: {cv_scores.round(4)}")
    print(f"   Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate on test set
    print("\n[5/5] Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    }
    
    print("\n" + "-" * 40)
    print("MODEL PERFORMANCE METRICS")
    print("-" * 40)
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\n" + "-" * 40)
    print("CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))
    
    print("-" * 40)
    print("CONFUSION MATRIX")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(f"   [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"    [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    # Feature importance
    print("\n" + "-" * 40)
    print("FEATURE IMPORTANCE")
    print("-" * 40)
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 40)
        print(f"   {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    train_model()
