#!/usr/bin/env python3
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from src.data.data_loader import GenomicDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import yaml
import pandas as pd
import numpy as np

def main():
    print("🧬 Small VCF Training Script")
    print("=" * 50)
    
    # Load config
    with open(project_root / 'config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load VCF data
    print("📊 Loading VCF data...")
    loader = GenomicDataLoader(config)
    vcf_data = loader.load_vcf(str(project_root / 'data/raw/clinvar_20250915.vcf.gz'))
    
    print(f"✅ Loaded {len(vcf_data)} variants")
    
    # Sample the data to make it manageable
    if len(vcf_data) > 50000:  # If more than 50k variants
        print(f"⚠️  Dataset too large ({len(vcf_data)} variants). Sampling 50,000 variants...")
        vcf_data = vcf_data.sample(n=50000, random_state=42)
        print(f"✅ Sampled {len(vcf_data)} variants for training")
    
    print(f"📋 Columns: {list(vcf_data.columns)}")
    
    # Check if we have pathogenicity labels
    if 'pathogenicity' not in vcf_data.columns:
        print("⚠️  No pathogenicity column found. Creating synthetic labels for demo.")
        vcf_data['pathogenicity'] = np.random.randint(0, 2, len(vcf_data))
    else:
        print("✅ Found pathogenicity column")
    
    # Prepare features - use only numeric columns
    print("�� Preparing features...")
    exclude_cols = ['pathogenicity', 'CHROM', 'REF', 'ALT', 'ID', 'QUAL', 'FILTER', 'INFO']
    feature_cols = [col for col in vcf_data.columns if col not in exclude_cols]
    
    # Only use numeric columns
    X = vcf_data[feature_cols].select_dtypes(include=[np.number])
    y = vcf_data['pathogenicity'].astype(int)
    
    print(f"📊 Features: {list(X.columns)}")
    print(f"�� Target distribution: {y.value_counts().to_dict()}")
    
    if len(X.columns) == 0:
        print("❌ No numeric features found. Creating synthetic features for demo.")
        X = pd.DataFrame({
            'feature_1': np.random.randn(len(vcf_data)),
            'feature_2': np.random.randn(len(vcf_data)),
            'feature_3': np.random.randn(len(vcf_data))
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
    
    # Train model with smaller parameters
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=10,     # Limit depth
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model accuracy: {accuracy:.3f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Show feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        top_features = sorted(zip(X.columns, feature_importance), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🔝 Top 10 features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i:2d}. {feature}: {importance:.3f}")
    
    # Save model
    print("💾 Saving model...")
    import joblib
    import os
    models_dir = project_root / 'models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, models_dir / 'small_model.pkl')
    joblib.dump(scaler, models_dir / 'small_scaler.pkl')
    print("✅ Model saved to models/small_model.pkl")
    print("✅ Scaler saved to models/small_scaler.pkl")
    
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
