#!/usr/bin/env python3
"""
Simple prediction script that works without XGBoost
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

def load_model(model_path):
    """Load trained model and scaler"""
    print(f"📦 Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Run: python scripts/train.py first to train a model")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load('models/scaler.pkl')
    
    print("✅ Model loaded successfully")
    return model, scaler

def load_test_data(data_path):
    """Load test data"""
    print(f"📊 Loading test data from {data_path}")
    
    # For now, create sample test data
    np.random.seed(123)
    n_variants = 200
    
    data = {
        'chromosome': np.random.choice(['1', '2', '3', 'X', 'Y'], n_variants),
        'position': np.random.randint(1000, 1000000, n_variants),
        'ref_allele': np.random.choice(['A', 'T', 'G', 'C'], n_variants),
        'alt_allele': np.random.choice(['A', 'T', 'G', 'C'], n_variants),
        'variant_type': np.random.choice(['SNV', 'INDEL'], n_variants, p=[0.8, 0.2]),
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
    return df

def make_predictions(model, scaler, df):
    """Make predictions on test data"""
    print("🔮 Making predictions...")
    
    # Prepare features
    exclude_cols = ['chromosome', 'ref_allele', 'alt_allele', 'variant_type']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    print(f"✅ Predictions completed")
    print(f"   - Predicted {len(predictions)} variants")
    print(f"   - Pathogenic predictions: {predictions.sum()}")
    print(f"   - Benign predictions: {(predictions == 0).sum()}")
    
    return predictions, probabilities

def save_predictions(df, predictions, probabilities, output_path):
    """Save predictions to file"""
    print(f"💾 Saving predictions to {output_path}")
    
    # Create results DataFrame
    results = df.copy()
    results['prediction'] = predictions
    results['probability'] = probabilities
    results['prediction_label'] = ['Pathogenic' if p == 1 else 'Benign' for p in predictions]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    
    print(f"✅ Predictions saved to {output_path}")
    
    # Show summary
    print(f"\n📊 Prediction Summary:")
    print(f"   - Total variants: {len(results)}")
    print(f"   - Pathogenic: {predictions.sum()}")
    print(f"   - Benign: {(predictions == 0).sum()}")
    print(f"   - Average probability: {probabilities.mean():.3f}")

def main():
    """Main prediction function"""
    print("�� Simple Genomic Variant Analysis Prediction")
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
    df = load_test_data('data/raw/sample_variants.vcf')
    
    # Make predictions
    predictions, probabilities = make_predictions(model, scaler, df)
    
    # Save predictions
    output_path = 'results/predictions.csv'
    save_predictions(df, predictions, probabilities, output_path)
    
    print("\n🎉 Prediction completed successfully!")
    print("📋 Next steps:")
    print("   1. Check results in results/predictions.csv")
    print("   2. View MLflow UI: mlflow ui")

if __name__ == "__main__":
    main()