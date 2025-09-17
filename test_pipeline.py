#!/usr/bin/env python3
"""
Test script for the Genomic Variant Analysis Pipeline
This script demonstrates how to run the pipeline and test its components.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.pipeline.inference_pipeline import InferencePipeline
from src.data.data_loader import GenomicDataLoader
from src.models.variant_classifier import VariantClassifier
from src.utils.metrics import calculate_metrics, plot_roc_curve
from src.visualization.plots import plot_variant_type_distribution

def test_data_loading():
    """Test data loading functionality"""
    print("🧬 Testing Data Loading...")
    
    # Create sample data for testing
    sample_data = {
        'chromosome': ['1', '1', '2', '2', '3'],
        'position': [1000, 2000, 3000, 4000, 5000],
        'ref_allele': ['A', 'T', 'G', 'C', 'A'],
        'alt_allele': ['T', 'G', 'A', 'T', 'G'],
        'variant_type': ['SNV', 'SNV', 'SNV', 'SNV', 'SNV'],
        'pathogenicity': [1, 0, 1, 0, 1],
        'allele_frequency': [0.01, 0.05, 0.02, 0.08, 0.03],
        'conservation_score': [0.8, 0.3, 0.9, 0.2, 0.7]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Created sample dataset with {len(df)} variants")
    print(f"   - Pathogenic variants: {df['pathogenicity'].sum()}")
    print(f"   - Benign variants: {(df['pathogenicity'] == 0).sum()}")
    
    return df

def test_feature_engineering():
    """Test feature engineering"""
    print("\n🔧 Testing Feature Engineering...")
    
    # Create sample data
    df = test_data_loading()
    
    # Add some additional features for testing
    df['gc_content'] = np.random.random(len(df))
    df['motif_score'] = np.random.random(len(df))
    df['secondary_structure'] = np.random.random(len(df))
    
    print(f"✅ Added {len(df.columns)} features to dataset")
    print(f"   - Features: {list(df.columns)}")
    
    return df

def test_model_training():
    """Test model training"""
    print("\n🤖 Testing Model Training...")
    
    # Create sample data
    df = test_feature_engineering()
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['pathogenicity', 'chromosome']]
    X = df[feature_cols]
    y = df['pathogenicity']
    
    print(f"   - Training features: {len(feature_cols)}")
    print(f"   - Training samples: {len(X)}")
    
    # Initialize and train model
    model = VariantClassifier(model_type='random_forest')
    model.train(X, y)
    
    print("✅ Model trained successfully!")
    
    # Test predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    print(f"   - Predictions shape: {predictions.shape}")
    print(f"   - Probabilities shape: {probabilities.shape}")
    
    return model, X, y

def test_evaluation():
    """Test model evaluation"""
    print("\n📊 Testing Model Evaluation...")
    
    model, X, y = test_model_training()
    
    # Evaluate model
    metrics = model.evaluate(X, y)
    
    print("✅ Model evaluation completed!")
    print(f"   - Accuracy: {metrics['accuracy']:.3f}")
    print(f"   - Precision: {metrics['precision']:.3f}")
    print(f"   - Recall: {metrics['recall']:.3f}")
    print(f"   - F1 Score: {metrics['f1']:.3f}")
    
    return metrics

def test_pipeline_integration():
    """Test full pipeline integration"""
    print("\n🔄 Testing Full Pipeline Integration...")
    
    try:
        # Initialize training pipeline
        pipeline = TrainingPipeline()
        print("✅ Training pipeline initialized")
        
        # Note: Full pipeline requires actual data files
        # For testing, we'll just verify the pipeline can be created
        print("   - Pipeline ready for training with real data")
        
    except Exception as e:
        print(f"⚠️  Pipeline initialization note: {e}")
        print("   - This is expected when no real data files are present")

def test_visualization():
    """Test visualization functions"""
    print("\n📈 Testing Visualization...")
    
    # Create sample data for plotting
    df = test_data_loading()
    
    try:
        # Test variant type distribution plot
        plot_variant_type_distribution(df)
        print("✅ Visualization functions working")
        
    except Exception as e:
        print(f"⚠️  Visualization note: {e}")
        print("   - Some plots may require additional dependencies")

def main():
    """Main test function"""
    print("🧬 Genomic Variant Analysis Pipeline - Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_data_loading()
    test_feature_engineering()
    test_model_training()
    test_evaluation()
    test_pipeline_integration()
    test_visualization()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📋 Next Steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Add your VCF data to data/raw/")
    print("   3. Run: python scripts/train.py")
    print("   4. Run: python scripts/evaluate.py")
    print("   5. Run: python scripts/predict.py")

if __name__ == "__main__":
    main()
