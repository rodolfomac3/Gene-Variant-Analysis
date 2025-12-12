#!/usr/bin/env python3
"""
Train model with REAL pathogenicity labels from ClinVar
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_config():
    """Load configuration"""
    with open(project_root / 'config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def extract_pathogenicity_labels(vcf_data):
    """Extract real pathogenicity labels from ClinVar INFO column"""
    print("🔬 Extracting real pathogenicity labels from ClinVar...")
    
    # Get the INFO column (last column)
    info_col = vcf_data.columns[-1]
    info_data = vcf_data[info_col]
    
    # Extract CLNSIG values (pathogenicity classification)
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
    label_mapping = {}
    
    for label in pathogenicity_labels:
        if label in ['Pathogenic', 'Likely_pathogenic']:
            binary_labels.append(1)  # Pathogenic
            label_mapping[label] = 1
        elif label in ['Benign', 'Likely_benign']:
            binary_labels.append(0)  # Benign
            label_mapping[label] = 0
        else:
            binary_labels.append(-1)  # Unknown/Uncertain
    
    # Create mapping for reference
    print(f"📊 Pathogenicity label distribution:")
    unique_labels = pd.Series(pathogenicity_labels).value_counts()
    for label, count in unique_labels.items():
        print(f"   - {label}: {count}")
    
    print(f"\\n🔢 Binary label mapping:")
    for label, binary in label_mapping.items():
        print(f"   - {label} → {binary}")
    
    return np.array(binary_labels), pathogenicity_labels, label_mapping

def main():
    """Main training function with real labels"""
    print("🧬 Training Model with REAL ClinVar Pathogenicity Labels")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Load VCF data
    print("📊 Loading VCF data...")
    from src.data.data_loader import GenomicDataLoader
    loader = GenomicDataLoader(config)
    vcf_data = loader.load_vcf(str(project_root / 'data/raw/clinvar_20250915.vcf.gz'))
    
    print(f"✅ Loaded {len(vcf_data)} variants")
    
    # Extract real pathogenicity labels
    binary_labels, original_labels, label_mapping = extract_pathogenicity_labels(vcf_data)
    
    # Filter out unknown labels
    known_mask = binary_labels != -1
    vcf_data_known = vcf_data[known_mask]
    binary_labels_known = binary_labels[known_mask]
    original_labels_known = [original_labels[i] for i in range(len(original_labels)) if known_mask[i]]
    
    print(f"\\n📊 After filtering unknown labels:")
    print(f"   - Total variants: {len(vcf_data_known)}")
    print(f"   - Pathogenic: {(binary_labels_known == 1).sum()}")
    print(f"   - Benign: {(binary_labels_known == 0).sum()}")
    
    # Sample data if too large
    if len(vcf_data_known) > 100000:
        print(f"⚠️  Dataset too large ({len(vcf_data_known)} variants). Sampling 100,000 variants...")
        sample_indices = np.random.choice(len(vcf_data_known), 100000, replace=False)
        vcf_data_known = vcf_data_known.iloc[sample_indices]
        binary_labels_known = binary_labels_known[sample_indices]
        original_labels_known = [original_labels_known[i] for i in sample_indices]
    
    # Prepare features
    print("🔧 Preparing features...")
    exclude_cols = ['pathogenicity', 'CHROM', 'REF', 'ALT', 'ID', 'QUAL', 'FILTER', 'INFO']
    feature_cols = [col for col in vcf_data_known.columns if col not in exclude_cols]
    
    # Only use numeric columns
    X = vcf_data_known[feature_cols].select_dtypes(include=[np.number])
    y = binary_labels_known
    
    print(f"📊 Features: {list(X.columns)}")
    print(f"📊 Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    if len(X.columns) == 0:
        print("❌ No numeric features found. Creating synthetic features for demo.")
        X = pd.DataFrame({
            'feature_1': np.random.randn(len(vcf_data_known)),
            'feature_2': np.random.randn(len(vcf_data_known)),
            'feature_3': np.random.randn(len(vcf_data_known))
        })
    
    # Split data
    print("🔄 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\\n📈 Model Performance:")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Show classification report
    print(f"\\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic']))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        top_features = sorted(zip(X.columns, feature_importance), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\\n🔝 Top 10 features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feature}: {importance:.3f}")
    
    # Save model
    print("💾 Saving model...")
    models_dir = project_root / 'models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, models_dir / 'real_model.pkl')
    joblib.dump(scaler, models_dir / 'real_scaler.pkl')
    
    # Save label mapping
    import json
    with open(models_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print("✅ Model saved to models/real_model.pkl")
    print("✅ Scaler saved to models/real_scaler.pkl")
    print("✅ Label mapping saved to models/label_mapping.json")
    
    print("\\n🎉 Training with REAL labels completed successfully!")
    print("\\n📋 Next steps:")
    print("   1. Evaluate: python scripts/evaluate_real.py")
    print("   2. Predict: python scripts/predict.py")

if __name__ == "__main__":
    main()


