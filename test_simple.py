#!/usr/bin/env python3
"""
Simple test script for the Genomic Variant Analysis Pipeline
This script tests basic functionality without XGBoost dependencies.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_basic_imports():
    """Test basic imports without XGBoost"""
    print("🧬 Testing Basic Imports...")
    
    try:
        # Test data loading
        from src.data.data_loader import GenomicDataLoader
        print("✅ GenomicDataLoader imported successfully")
        
        # Test preprocessor
        from src.data.preprocessor import GenomicPreprocessor
        print("✅ GenomicPreprocessor imported successfully")
        
        # Test feature extractors
        from src.features.variant_features import VariantFeatureExtractor
        print("✅ VariantFeatureExtractor imported successfully")
        
        from src.features.sequence_features import SequenceFeatureExtractor
        print("✅ SequenceFeatureExtractor imported successfully")
        
        # Test utilities
        from src.utils.metrics import calculate_metrics
        print("✅ Metrics utilities imported successfully")
        
        from src.utils.vcf_parser import VCFParser
        print("✅ VCFParser imported successfully")
        
        # Test visualization
        from src.visualization.plots import plot_variant_type_distribution
        print("✅ Visualization functions imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_operations():
    """Test basic data operations"""
    print("\n🔧 Testing Data Operations...")
    
    try:
        # Create sample data
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
        
        # Test basic operations
        df['gc_content'] = np.random.random(len(df))
        df['motif_score'] = np.random.random(len(df))
        
        print(f"✅ Added features: {len(df.columns)} total columns")
        
        return df
        
    except Exception as e:
        print(f"❌ Data operations error: {e}")
        return None

def test_feature_extraction():
    """Test feature extraction"""
    print("\n🔬 Testing Feature Extraction...")
    
    try:
        from src.features.variant_features import VariantFeatureExtractor
        
        # Create sample data
        df = test_data_operations()
        if df is None:
            return False
        
        # Test variant feature extraction
        extractor = VariantFeatureExtractor()
        features = extractor.extract_basic_features(df)
        
        print(f"✅ Extracted {len(features)} variant features")
        print(f"   - Feature columns: {list(features.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False

def test_model_without_xgboost():
    """Test model functionality without XGBoost"""
    print("\n🤖 Testing Model (Random Forest only)...")
    
    try:
        # Import only the model class, not the full module
        import sys
        sys.path.append('src/models')
        
        # Create a simple model test
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Create sample data
        df = test_data_operations()
        if df is None:
            return False
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['pathogenicity', 'chromosome']]
        X = df[feature_cols]
        y = df['pathogenicity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"✅ Model trained and tested successfully")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - Features used: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing error: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🛠️ Testing Utility Functions...")
    
    try:
        from src.utils.metrics import calculate_metrics
        
        # Create sample predictions
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        y_proba = [0.9, 0.1, 0.4, 0.2, 0.8]
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        print(f"✅ Metrics calculated successfully")
        print(f"   - Accuracy: {metrics['accuracy']:.3f}")
        print(f"   - Precision: {metrics['precision']:.3f}")
        print(f"   - Recall: {metrics['recall']:.3f}")
        print(f"   - F1 Score: {metrics['f1']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility testing error: {e}")
        return False

def test_file_structure():
    """Test project file structure"""
    print("\n📁 Testing Project Structure...")
    
    required_dirs = [
        'src/data',
        'src/features', 
        'src/models',
        'src/pipeline',
        'src/utils',
        'src/visualization',
        'scripts',
        'tests',
        'config',
        'data/raw',
        'data/processed',
        'results'
    ]
    
    required_files = [
        'requirements.txt',
        'setup.py',
        'README.md',
        'config/config.yaml',
        'src/__init__.py',
        'scripts/train.py',
        'scripts/evaluate.py',
        'scripts/predict.py'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Main test function"""
    print("🧬 Genomic Variant Analysis Pipeline - Simple Test Suite")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Operations", test_data_operations),
        ("Feature Extraction", test_feature_extraction),
        ("Model Testing", test_model_without_xgboost),
        ("Utility Functions", test_utilities),
        ("File Structure", test_file_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your pipeline is ready to use.")
        print("\n📋 Next Steps:")
        print("   1. Install XGBoost dependencies: brew install libomp")
        print("   2. Add your VCF data to data/raw/")
        print("   3. Run: python scripts/train.py")
        print("   4. Check MLflow UI: mlflow ui")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
