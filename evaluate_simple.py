#!/usr/bin/env python3
"""
Simple evaluation script that works without XGBoost
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_model(model_path):
    """Load trained model and scaler"""
    print(f"📦 Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load('models/scaler.pkl')
    
    print("✅ Model loaded successfully")
    return model, scaler

def load_test_data():
    """Load test data with known labels"""
    print("📊 Loading test data with labels...")
    
    # Create test data with known labels
    np.random.seed(456)
    n_variants = 300
    
    data = {
        'chromosome': np.random.choice(['1', '2', '3', 'X', 'Y'], n_variants),
        'position': np.random.randint(1000, 1000000, n_variants),
        'ref_allele': np.random.choice(['A', 'T', 'G', 'C'], n_variants),
        'alt_allele': np.random.choice(['A', 'T', 'G', 'C'], n_variants),
        'variant_type': np.random.choice(['SNV', 'INDEL'], n_variants, p=[0.8, 0.2]),
        'pathogenicity': np.random.choice([0, 1], n_variants, p=[0.7, 0.3]),  # True labels
        'allele_frequency': np.random.beta(1, 9, n_variants),
        'conservation_score': np.random.uniform(0, 1, n_variants),
        'gc_content': np.random.uniform(0.3, 0.7, n_variants),
        'motif_score': np.random.uniform(0, 1, n_variants),
        'secondary_structure': np.random.uniform(0, 1, n_variants)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure ref != alt for SNVs
    snv_mask = df['variant_type'] == 'SNV'
    same_allele = df.loc[snv_mask, 'ref_allele'] == df.loc[snv_mask, 'alt_allele']
    if same_allele.any():
        df.loc[snv_mask & same_allele, 'alt_allele'] = np.random.choice(['A', 'T', 'G', 'C'], same_allele.sum())
    
    print(f"✅ Loaded {len(df)} test variants")
    print(f"   - Pathogenic: {df['pathogenicity'].sum()}")
    print(f"   - Benign: {(df['pathogenicity'] == 0).sum()}")
    
    return df

def evaluate_model(model, scaler, df):
    """Evaluate model performance"""
    print("📊 Evaluating model performance...")
    
    # Prepare features
    exclude_cols = ['pathogenicity', 'chromosome', 'ref_allele', 'alt_allele', 'variant_type']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
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
    print(f"\n�� Model Performance:")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Precision: {precision:.3f}")
    print(f"   - Recall: {recall:.3f}")
    print(f"   - F1 Score: {f1:.3f}")
    print(f"   - AUC: {auc:.3f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n�� Confusion Matrix:")
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
    print("�� Simple Genomic Variant Analysis Evaluation")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'models/simple_model.pkl'
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("   Run: python scripts/train.py")
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
    print("   1. Check results in results/ directory")
    print("   2. View MLflow UI: mlflow ui")

if __name__ == "__main__":
    main()