"""
Evaluation pipeline for genomic variant analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mlflow
import mlflow.sklearn
from loguru import logger
import yaml
import json
from datetime import datetime

from ..data.data_loader import GenomicDataLoader
from ..data.preprocessor import GenomicPreprocessor
from ..models.variant_classifier import VariantClassifier
from ..models.ensemble_model import EnsembleVariantClassifier
from ..utils.metrics import calculate_metrics, plot_metrics, calculate_feature_importance_metrics

class EvaluationPipeline:
    """Evaluation pipeline for model assessment"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = GenomicDataLoader(self.config)
        self.preprocessor = GenomicPreprocessor(self.config)
        
        # MLflow setup
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def evaluate_model(
        self,
        model_path: str,
        test_data_path: str,
        output_path: Optional[str] = None,
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test data
            output_path: Path to save evaluation results
            generate_plots: Whether to generate evaluation plots
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting model evaluation")
        
        # Load model
        model = self._load_model(model_path)
        
        # Load test data
        test_data = self._load_test_data(test_data_path)
        
        # Prepare data
        X_test, y_test = self._prepare_test_data(test_data)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate plots if requested
        if generate_plots:
            plot_path = output_path.replace('.json', '_plots.png') if output_path else None
            plot_metrics(y_test, y_pred, y_pred_proba, save_path=plot_path)
        
        # Calculate feature importance metrics if available
        feature_importance_metrics = {}
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            feature_importance_metrics = calculate_feature_importance_metrics(model.feature_importance)
        
        # Create evaluation results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "test_data_path": test_data_path,
            "metrics": metrics,
            "feature_importance_metrics": feature_importance_metrics,
            "n_test_samples": len(X_test),
            "n_features": len(X_test.columns)
        }
        
        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("test_data_path", test_data_path)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            if output_path:
                mlflow.log_artifact(output_path)
        
        logger.info("Model evaluation completed successfully")
        return results
    
    def _load_model(self, model_path: str):
        """Load trained model from file."""
        logger.info(f"Loading model from {model_path}")
        
        try:
            if 'ensemble' in model_path.lower():
                model = EnsembleVariantClassifier.load(model_path)
            else:
                model = VariantClassifier.load(model_path)
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_test_data(self, test_data_path: str) -> pd.DataFrame:
        """Load test data from file."""
        logger.info(f"Loading test data from {test_data_path}")
        
        try:
            if test_data_path.endswith('.csv'):
                data = pd.read_csv(test_data_path)
            elif test_data_path.endswith('.parquet'):
                data = pd.read_parquet(test_data_path)
            else:
                data = pd.read_csv(test_data_path)
            
            logger.info(f"Test data loaded: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def _prepare_test_data(self, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare test data for evaluation."""
        logger.info("Preparing test data for evaluation")
        
        # Assume target column is 'pathogenicity' or similar
        target_col = 'pathogenicity'
        if target_col not in test_data.columns:
            # Try to find target column
            possible_targets = ['target', 'label', 'y', 'pathogenicity', 'disease']
            target_col = None
            for col in possible_targets:
                if col in test_data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError(f"Target column not found. Available columns: {test_data.columns.tolist()}")
        
        # Separate features and target
        feature_cols = [col for col in test_data.columns if col != target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Apply preprocessing if preprocessor is available
        # Note: In practice, you'd load the fitted preprocessor
        # For now, we'll assume the data is already preprocessed
        
        logger.info(f"Test data prepared: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test
