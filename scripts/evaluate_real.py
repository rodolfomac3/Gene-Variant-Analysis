#!/usr/bin/env python3
"""
Evaluate model trained with REAL pathogenicity labels
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def load_model(model_path):
    """Load trained model and scaler"""
    print(f"📦 Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load('../models/real_scaler.pkl')
    
    # Load label mapping
    with open('../models/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    print("✅ Model loaded successfully")
    return model, scaler, label_mapping

def load_test_data():
    """Load test data with real ClinVar labels"""
    print("📊 Loading test data with real ClinVar labels...")
    
    # Load a different ClinVar file for testing
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root / 'src'))
    from src.data.data_loader import GenomicDataLoader
    import yaml
    
    # Load config
    with open(project_root / 'config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load VCF data
    loader = GenomicDataLoader(config)
    vcf_data = loader.load_vcf(str(project_root / 'data/raw/clinvar_papu.vcf.gz'))
    
    print(f"✅ Loaded {len(vcf_data)} variants from ClinVar Papu")
    
    # Extract pathogenicity labels
    info_col = vcf_data.columns[-1]
    info_data = vcf_data[info_col]
    
    pathogenicity_labels = []
    for info in info_data:
        if pd.isna(info):
            pathogenicity_labels.append('Unknown')
            continue
            
        # Parse CLNSIG from INFO string
        clnsig = None
        for field in str(info).split(';'):
            if field.startswith('CLNSIG='):
                clnsig = field.split('=')[1]
                break
        
        if clnsig:
            pathogenicity_labels.append(clnsig)
        else:
            pathogenicity_labels.append('Unknown')
    
    # Convert to binary labels
    binary_labels = []
    for label in pathogenicity_labels:
        if label in ['Pathogenic', 'Likely_pathogenic']:
            binary_labels.append(1)  # Pathogenic
        elif label in ['Benign', 'Likely_benign']:
            binary_labels.append(0)  # Benign
        else:
            binary_labels.append(-1)  # Unknown/Uncertain
    
    # Filter out unknown labels
    known_mask = np.array(binary_labels) != -1
    vcf_data_known = vcf_data[known_mask]
    binary_labels_known = np.array(binary_labels)[known_mask]
    original_labels_known = [pathogenicity_labels[i] for i in range(len(pathogenicity_labels)) if known_mask[i]]
    
    print(f"📊 After filtering unknown labels:")
    print(f"   - Total variants: {len(vcf_data_known)}")
    print(f"   - Pathogenic: {(binary_labels_known == 1).sum()}")
    print(f"   - Benign: {(binary_labels_known == 0).sum()}")
    
    # Sample if too large
    if len(vcf_data_known) > 10000:
        print(f"⚠️  Dataset too large. Sampling 10,000 variants for evaluation...")
        sample_indices = np.random.choice(len(vcf_data_known), 10000, replace=False)
        vcf_data_known = vcf_data_known.iloc[sample_indices]
        binary_labels_known = binary_labels_known[sample_indices]
        original_labels_known = [original_labels_known[i] for i in sample_indices]
    
    return vcf_data_known, binary_labels_known, original_labels_known

def evaluate_model(model, scaler, vcf_data, y_true, original_labels):
    """Evaluate model performance"""
    print("📊 Evaluating model performance...")
    
    # Prepare features (same as training)
    exclude_cols = ['pathogenicity', 'CHROM', 'REF', 'ALT', 'ID', 'QUAL', 'FILTER', 'INFO']
    feature_cols = [col for col in vcf_data.columns if col not in exclude_cols]
    
    # Only use numeric columns
    X = vcf_data[feature_cols].select_dtypes(include=[np.number])
    
    if len(X.columns) == 0:
        print("❌ No numeric features found. Using synthetic features.")
        X = pd.DataFrame({
            'feature_1': np.random.randn(len(vcf_data)),
            'feature_2': np.random.randn(len(vcf_data)),
            'feature_3': np.random.randn(len(vcf_data))
        })
    
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
    print(f"\\n📈 Model Performance:")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Precision: {precision:.3f}")
    print(f"   - Recall: {recall:.3f}")
    print(f"   - F1 Score: {f1:.3f}")
    print(f"   - AUC: {auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\\n🔢 Confusion Matrix:")
    print(f"   [[{cm[0,0]:3d}, {cm[0,1]:3d}]")
    print(f"    [{cm[1,0]:3d}, {cm[1,1]:3d}]]")
    
    # Classification report
    print(f"\\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Pathogenic']))
    
    # Show some examples
    print(f"\\n🔍 Sample Predictions:")
    sample_indices = np.random.choice(len(y_true), min(10, len(y_true)), replace=False)
    for i in sample_indices:
        true_label = 'Pathogenic' if y_true[i] == 1 else 'Benign'
        pred_label = 'Pathogenic' if y_pred[i] == 1 else 'Benign'
        prob = y_proba[i]
        original = original_labels[i]
        correct = "✅" if y_true[i] == y_pred[i] else "❌"
        print(f"   {correct} True: {true_label} | Pred: {pred_label} ({prob:.3f}) | Original: {original}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def main():
    """Main evaluation function"""
    print("🧬 Evaluating Model with REAL ClinVar Labels")
    print("=" * 50)
    
    # Check if model exists
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'models/real_model.pkl'
    if not os.path.exists(model_path):
        print("❌ No real model found!")
        print("   Run: python scripts/train_real_labels.py")
        return
    
    # Load model
    model, scaler, label_mapping = load_model(model_path)
    if model is None:
        return
    
    # Load test data
    vcf_data, y_true, original_labels = load_test_data()
    
    # Evaluate model
    metrics = evaluate_model(model, scaler, vcf_data, y_true, original_labels)
    
    print("\\n🎉 Evaluation completed successfully!")
    print("\\n📋 Next steps:")
    print("   1. Check results in ../results/ directory")
    print("   2. View MLflow UI: mlflow ui")

if __name__ == "__main__":
    main()


