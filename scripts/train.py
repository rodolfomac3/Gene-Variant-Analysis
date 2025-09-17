#!/usr/bin/env python3
"""
Training script for genomic variant analysis pipeline
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.training_pipeline import TrainingPipeline
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Train genomic variant classifier")
    parser.add_argument("--vcf", required=True, help="Path to VCF file")
    parser.add_argument("--annotations", help="Path to annotations file")
    parser.add_argument("--model-type", default="random_forest", 
                       choices=["xgboost", "lightgbm", "random_forest", "gradient_boosting", "logistic_regression"],
                       help="Type of model to train")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of tuning trials")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = TrainingPipeline(config_path=args.config)
        
        # Run training
        results = pipeline.run(
            vcf_path=args.vcf,
            annotation_path=args.annotations,
            model_type=args.model_type,
            use_ensemble=args.ensemble
        )
        
        # Print results
        logger.info(f"Training completed successfully!")
        logger.info(f"Run ID: {results['run_id']}")
        logger.info(f"Test AUC: {results['metrics'].get('test_auc', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
