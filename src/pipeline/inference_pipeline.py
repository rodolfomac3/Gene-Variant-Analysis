"""
Inference pipeline for genomic variant analysis.
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
from ..features.variant_features import VariantFeatureExtractor
from ..features.sequence_features import SequenceFeatureExtractor
from ..features.annotation_features import AnnotationFeatureExtractor
from ..models.variant_classifier import VariantClassifier
from ..models.ensemble_model import EnsembleVariantClassifier

class InferencePipeline:
    """Inference pipeline for making predictions on new data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = GenomicDataLoader(self.config)
        self.preprocessor = GenomicPreprocessor(self.config)
        self.variant_feature_extractor = VariantFeatureExtractor(self.config)
        self.sequence_feature_extractor = SequenceFeatureExtractor(self.config)
        self.annotation_feature_extractor = AnnotationFeatureExtractor(self.config)
        
        # MLflow setup
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def predict(
        self,
        model_path: str,
        data_path: str,
        output_path: Optional[str] = None,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            model_path: Path to trained model
            data_path: Path to input data
            output_path: Path to save predictions
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Starting inference pipeline")
        
        # Load model
        model = self._load_model(model_path)
        
        # Load data
        data = self._load_data(data_path)
        
        # Extract features
        features = self._extract_features(data)
        
        # Make predictions
        predictions = self._make_predictions(model, features, return_probabilities)
        
        # Combine with original data
        result_df = data.copy()
        result_df['prediction'] = predictions['prediction']
        
        if return_probabilities and 'probability' in predictions:
            result_df['prediction_probability'] = predictions['probability']
        
        # Save results
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("data_path", data_path)
            mlflow.log_metric("n_predictions", len(result_df))
            
            if output_path:
                mlflow.log_artifact(output_path)
        
        logger.info("Inference pipeline completed successfully")
        return result_df
    
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
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load input data from file."""
        logger.info(f"Loading data from {data_path}")
        
        try:
            if data_path.endswith('.vcf'):
                data = self.data_loader.load_vcf(data_path)
            elif data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                data = pd.read_csv(data_path)
            
            logger.info(f"Data loaded: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from input data."""
        logger.info("Extracting features")
        
        # Extract variant features
        features = self.variant_feature_extractor.extract_basic_features(data)
        
        # Extract sequence features if sequence data is available
        if 'sequence' in data.columns:
            sequences = data['sequence'].tolist()
            sequence_features = self.sequence_feature_extractor.extract_motif_features(sequences)
            features = pd.concat([features, sequence_features], axis=1)
        
        # Extract annotation features if available
        if any(col in data.columns for col in ['gene_name', 'consequence', 'impact']):
            features = self.annotation_feature_extractor.extract_gene_features(features)
        
        logger.info(f"Features extracted: {features.shape}")
        return features
    
    def _make_predictions(self, model, features: pd.DataFrame, 
                         return_probabilities: bool = False) -> Dict[str, np.ndarray]:
        """Make predictions using the trained model."""
        logger.info("Making predictions")
        
        # Ensure features are in the correct format
        # Remove any non-numeric columns that might cause issues
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Make predictions
        predictions = model.predict(numeric_features)
        
        result = {'prediction': predictions}
        
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(numeric_features)
            result['probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        logger.info(f"Predictions made: {len(predictions)} samples")
        return result