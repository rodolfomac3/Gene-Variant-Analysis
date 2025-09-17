import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from loguru import logger
import joblib

from ..data.preprocessor import GenomicPreprocessor
from ..features.variant_features import VariantFeatureExtractor
from ..models.variant_classifier import VariantClassifier

class InferencePipeline:
    """Pipeline for making predictions on new data"""
    
    def __init__(self, model_path: str, preprocessor_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to fitted preprocessor
            config_path: Path to config file
        """
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model and preprocessor
        self.model = VariantClassifier.load(model_path)
        self.preprocessor = GenomicPreprocessor.load(preprocessor_path)
        self.feature_extractor = VariantFeatureExtractor(self.config)
        
        logger.info("Inference pipeline initialized")
    
    def predict_single(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single variant
        
        Args:
            variant: Dictionary with variant information
            
        Returns:
            Prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([variant])
        
        # Make predictions
        results = self.predict_batch(df)
        
        return results[0]
    
    def predict_batch(self, variants_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple variants
        
        Args:
            variants_df: DataFrame with variants
            
        Returns:
            List of prediction results
        """
        logger.info(f"Making predictions for {len(variants_df)} variants")
        
        # Extract features
        features_df = self.feature_extractor.extract_basic_features(variants_df)
        features_df = self.feature_extractor.extract_conservation_features(features_df)
        features_df = self.feature_extractor.extract_population_features(features_df)
        
        # Preprocess
        features_df = self.preprocessor.handle_missing_values(features_df)
        features_df = self.preprocessor.transform(features_df)
        
        # Make predictions
        predictions = self.model.predict(features_df)
        probabilities = self.model.predict_proba(features_df)
        
        # Format results
        results = []
        for i, (_, row) in enumerate(variants_df.iterrows()):
            result = {
                'variant': f"{row['chrom']}:{row['pos']}:{row['ref']}>{row['alt']}",
                'prediction': int(predictions[i]),
                'probability': float(probabilities[i, 1]),
                'confidence': float(max(probabilities[i])),
                'classification': 'pathogenic' if predictions[i] == 1 else 'benign'
            }
            results.append(result)
        
        logger.info("Predictions completed")
        
        return results
    
    def predict_from_vcf(self, vcf_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions from VCF file
        
        Args:
            vcf_path: Path to VCF file
            output_path: Optional path to save results
            
        Returns:
            DataFrame with predictions
        """
        from ..data.data_loader import GenomicDataLoader
        
        logger.info(f"Loading variants from {vcf_path}")
        
        # Load variants
        data_loader = GenomicDataLoader(self.config)
        variants_df = data_loader.load_vcf(vcf_path)
        
        # Make predictions
        results = self.predict_batch(variants_df)
        
        # Combine with original data
        results_df = pd.DataFrame(results)
        combined_df = pd.concat([variants_df, results_df], axis=1)
        
        # Save results if path provided
        if output_path:
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved results to {output_path}")
        
        return combined_df