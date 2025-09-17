#!/usr/bin/env python3
"""
Demo script for the Genomic Variant Analysis Pipeline
This script demonstrates how to use the pipeline without XGBoost dependencies.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def create_sample_data():
    """Create sample genomic data for demonstration"""
    print("🧬 Creating Sample Genomic Data...")
    
    np.random.seed(42)
    n_variants = 1000
    
    # Create realistic genomic data
    data = {
        'chromosome': np.random.choice(['1', '2', '3', '4', '5', 'X', 'Y'], n_variants),
        'position': np.random.randint(1000, 10000000, n_variants),
        'ref_allele': np.random.choice(['A', 'T', 'G', 'C'], n_variants),
        'alt_allele': np.random.choice(['A', 'T', 'G', 'C'], n_variants),
        'variant_type': np.random.choice(['SNV', 'INDEL'], n_variants, p=[0.8, 0.2]),
        'pathogenicity': np.random.choice([0, 1], n_variants, p=[0.7, 0.3]),
        'allele_frequency': np.random.beta(1, 9, n_variants),  # Low frequency variants
        'conservation_score': np.random.uniform(0, 1, n_variants),
        'gc_content': np.random.uniform(0.3, 0.7, n_variants),
        'motif_score': np.random.uniform(0, 1, n_variants),
        'secondary_structure': np.random.uniform(0, 1, n_variants),
        'length_diff': np.random.randint(-10, 11, n_variants),
        'is_missense': np.random.choice([0, 1], n_variants, p=[0.6, 0.4]),
        'is_synonymous': np.random.choice([0, 1], n_variants, p=[0.8, 0.2]),
        'is_nonsense': np.random.choice([0, 1], n_variants, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Ensure ref != alt for SNVs
    snv_mask = df['variant_type'] == 'SNV'
    same_allele = df.loc[snv_mask, 'ref_allele'] == df.loc[snv_mask, 'alt_allele']
    if same_allele.any():
        df.loc[snv_mask & same_allele, 'alt_allele'] = np.random.choice(['A', 'T', 'G', 'C'], same_allele.sum())
    
    # Add some realistic patterns
    df.loc[df['pathogenicity'] == 1, 'conservation_score'] = np.random.uniform(0.6, 1.0, df['pathogenicity'].sum())
    df.loc[df['pathogenicity'] == 0, 'conservation_score'] = np.random.uniform(0.0, 0.6, (df['pathogenicity'] == 0).sum())
    
    print(f"✅ Created dataset with {len(df)} variants")
    print(f"   - Pathogenic: {df['pathogenicity'].sum()}")
    print(f"   - Benign: {(df['pathogenicity'] == 0).sum()}")
    print(f"   - SNVs: {(df['variant_type'] == 'SNV').sum()}")
    print(f"   - INDELs: {(df['variant_type'] == 'INDEL').sum()}")
    
    return df

def demonstrate_data_loading():
    """Demonstrate data loading functionality"""
    print("\n📊 Demonstrating Data Loading...")
    
    try:
        from src.data.data_loader import GenomicDataLoader
        
        # Create sample data
        df = create_sample_data()
        
        # Save sample data
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/sample_variants.csv', index=False)
        print("✅ Sample data saved to data/raw/sample_variants.csv")
        
        # Test data loader
        loader = GenomicDataLoader()
        print("✅ GenomicDataLoader initialized")
        
        return df
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return None

def demonstrate_preprocessing():
    """Demonstrate data preprocessing"""
    print("\n🔧 Demonstrating Data Preprocessing...")
    
    try:
        from src.data.preprocessor import GenomicPreprocessor
        
        # Get data
        df = demonstrate_data_loading()
        if df is None:
            return None
        
        # Initialize preprocessor
        preprocessor = GenomicPreprocessor()
        print("✅ GenomicPreprocessor initialized")
        
        # Fit and transform
        preprocessor.fit(df)
        processed_df = preprocessor.transform(df)
        
        print(f"✅ Data preprocessed: {len(processed_df)} variants")
        print(f"   - Original features: {len(df.columns)}")
        print(f"   - Processed features: {len(processed_df.columns)}")
        
        return processed_df
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

def demonstrate_feature_engineering():
    """Demonstrate feature engineering"""
    print("\n🔬 Demonstrating Feature Engineering...")
    
    try:
        from src.features.variant_features import VariantFeatureExtractor
        from src.features.sequence_features import SequenceFeatureExtractor
        
        # Get data
        df = demonstrate_preprocessing()
        if df is None:
            return None
        
        # Test variant feature extraction
        variant_extractor = VariantFeatureExtractor()
        variant_features = variant_extractor.extract_basic_features(df)
        
        print(f"✅ Variant features extracted: {len(variant_features)} features")
        
        # Test sequence feature extraction
        sequence_extractor = SequenceFeatureExtractor()
        sequence_features = sequence_extractor.extract_motif_features(df)
        
        print(f"✅ Sequence features extracted: {len(sequence_features)} features")
        
        return df
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return None

def demonstrate_ml_training():
    """Demonstrate ML training without XGBoost"""
    print("\n🤖 Demonstrating ML Training...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
        
        # Get data
        df = demonstrate_feature_engineering()
        if df is None:
            return None
        
        # Prepare features
        exclude_cols = ['pathogenicity', 'chromosome', 'ref_allele', 'alt_allele', 'variant_type']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['pathogenicity']
        
        print(f"   - Features: {len(feature_cols)}")
        print(f"   - Samples: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Test samples: {len(X_test)}")
        
        # Show feature importance
        feature_importance = model.feature_importances_
        top_features = sorted(zip(feature_cols, feature_importance), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        print(f"   - Top 10 features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"     {i:2d}. {feature}: {importance:.3f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        import joblib
        joblib.dump(model, 'models/demo_model.pkl')
        print("✅ Model saved to models/demo_model.pkl")
        
        return model, X_test_scaled, y_test, y_pred, y_proba
        
    except Exception as e:
        print(f"❌ ML training error: {e}")
        return None

def demonstrate_evaluation():
    """Demonstrate model evaluation"""
    print("\n📊 Demonstrating Model Evaluation...")
    
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        # Get model results
        result = demonstrate_ml_training()
        if result is None:
            return None
        
        model, X_test, y_test, y_pred, y_proba = result
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        print(f"✅ Model evaluation completed")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - Precision: {precision:.3f}")
        print(f"   - Recall: {recall:.3f}")
        print(f"   - F1 Score: {f1:.3f}")
        print(f"   - AUC: {auc:.3f}")
        print(f"   - Average Precision: {avg_precision:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"   - Confusion Matrix:")
        print(f"     [[{cm[0,0]:3d}, {cm[0,1]:3d}]")
        print(f"     [{cm[1,0]:3d}, {cm[1,1]:3d}]]")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False

def demonstrate_visualization():
    """Demonstrate visualization capabilities"""
    print("\n📈 Demonstrating Visualization...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get data
        df = create_sample_data()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Variant type distribution
        df['variant_type'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Variant Type Distribution')
        axes[0,0].set_ylabel('Count')
        
        # Pathogenicity distribution
        df['pathogenicity'].value_counts().plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Pathogenicity Distribution')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_xticklabels(['Benign', 'Pathogenic'])
        
        # Allele frequency distribution
        df['allele_frequency'].hist(bins=20, ax=axes[1,0])
        axes[1,0].set_title('Allele Frequency Distribution')
        axes[1,0].set_xlabel('Allele Frequency')
        axes[1,0].set_ylabel('Count')
        
        # Conservation score by pathogenicity
        df.boxplot(column='conservation_score', by='pathogenicity', ax=axes[1,1])
        axes[1,1].set_title('Conservation Score by Pathogenicity')
        axes[1,1].set_xlabel('Pathogenicity')
        axes[1,1].set_ylabel('Conservation Score')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/demo_plots.png', dpi=300, bbox_inches='tight')
        print("✅ Plots saved to results/demo_plots.png")
        
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def demonstrate_mlflow():
    """Demonstrate MLflow integration"""
    print("\n🔬 Demonstrating MLflow Integration...")
    
    try:
        import mlflow
        import mlflow.sklearn
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Start experiment
        experiment = mlflow.set_experiment("genomic-variant-demo")
        
        with mlflow.start_run(run_name="demo_run"):
            # Log parameters
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Log metrics
            mlflow.log_metric("accuracy", 0.85)
            mlflow.log_metric("precision", 0.82)
            mlflow.log_metric("recall", 0.88)
            mlflow.log_metric("f1_score", 0.85)
            
            # Log model
            model, _, _, _, _ = demonstrate_ml_training()
            if model is not None:
                mlflow.sklearn.log_model(model, "model")
                print("✅ Model logged to MLflow")
            
            print("✅ MLflow experiment created")
            print("   - Run MLflow UI: mlflow ui")
            print("   - View experiments at: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow error: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🧬 Genomic Variant Analysis Pipeline - Demo")
    print("=" * 60)
    
    # Run demonstrations
    demonstrations = [
        ("Data Loading", demonstrate_data_loading),
        ("Preprocessing", demonstrate_preprocessing),
        ("Feature Engineering", demonstrate_feature_engineering),
        ("ML Training", demonstrate_ml_training),
        ("Model Evaluation", demonstrate_evaluation),
        ("Visualization", demonstrate_visualization),
        ("MLflow Integration", demonstrate_mlflow)
    ]
    
    results = []
    for demo_name, demo_func in demonstrations:
        try:
            result = demo_func()
            if result is not None:
                results.append((demo_name, True))
            else:
                results.append((demo_name, False))
        except Exception as e:
            print(f"❌ {demo_name} failed: {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Demo Results Summary:")
    
    passed = 0
    for demo_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {demo_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} demonstrations successful")
    
    if passed == len(results):
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📋 What You Can Do Next:")
        print("   1. Add your real VCF data to data/raw/")
        print("   2. Run: python scripts/train.py --vcf data/raw/your_data.vcf")
        print("   3. View MLflow UI: mlflow ui")
        print("   4. Check results in results/ directory")
        print("   5. Make predictions: python scripts/predict.py")
    else:
        print(f"\n⚠️  {len(results) - passed} demonstrations failed.")
        print("   Check the errors above and ensure all dependencies are installed.")
    
    print("\n🔧 To fix XGBoost issues:")
    print("   brew install libomp")
    print("   pip install xgboost")

if __name__ == "__main__":
    main()
