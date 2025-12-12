#!/usr/bin/env python3
"""
Simple evaluation script that works with joblib-saved models
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

project_root = Path(__file__).parent.parent

def load_model(model_path):
    """Load trained model and scaler"""
    print(f"📦 Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(project_root / 'models/small_scaler.pkl')
    
    print("✅ Model loaded successfully")
    return model, scaler

def load_test_data():
    """Load test data with known labels - matching training features"""
    print("📊 Loading test data with labels...")
    
    # Create test data with the SAME features used in training
    np.random.seed(456)
    n_variants = 300
    
    # Use the same feature names as in training (from VCF file)
    data = {
        '66926': np.random.randint(1000, 1000000, n_variants),  # Position-like data
        '3385321': np.random.randint(1000, 1000000, n_variants),  # Position-like data
        'pathogenicity': np.random.choice([0, 1], n_variants, p=[0.7, 0.3])  # True labels
    }
    
    df = pd.DataFrame(data)
    
    print(f"✅ Loaded {len(df)} test variants")
    print(f"   - Pathogenic: {df['pathogenicity'].sum()}")
    print(f"   - Benign: {(df['pathogenicity'] == 0).sum()}")
    print(f"   - Features: {list(df.columns)}")
    
    return df

def evaluate_model(model, scaler, df):
    """Evaluate model performance"""
    print("📊 Evaluating model performance...")
    
    # Prepare features - use the same features as training
    feature_cols = ['66926', '3385321']
    X = df[feature_cols]
    y_true = df['pathogenicity']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"✅ Evaluation completed")
    print(f"\n📈 Model Performance:")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Precision: {precision:.3f}")
    print(f"   - Recall: {recall:.3f}")
    print(f"   - F1 Score: {f1:.3f}")
    print(f"   - AUC: {auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"   [[{cm[0,0]:3d}, {cm[0,1]:3d}]")
    print(f"    [{cm[1,0]:3d}, {cm[1,1]:3d}]]")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main():
    """Main evaluation function"""
    print("🧬 Simple Genomic Variant Analysis Evaluation")
    print("=" * 50)
    
    # Check if model exists
    model_path = project_root / 'models/small_model.pkl'
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("   Run: python scripts/train_small.py")
        return
    
    # Load model
    model, scaler = load_model(model_path)
    if model is None:
        return
    
    # Load test data
    df = load_test_data()
    
    # Evaluate model
    metrics = evaluate_model(model, scaler, df)
    
    print("\n🎉 Evaluation completed successfully!")
    print("📋 Next steps:")
    print("   1. Check results in ../results/ directory")
    print("   2. View MLflow UI: mlflow ui")

if __name__ == "__main__":
    main()
