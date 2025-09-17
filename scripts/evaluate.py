#!/usr/bin/env python
"""
Evaluation script for genomic variant analysis pipeline
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.evaluation_pipeline import EvaluationPipeline
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Evaluate genomic variant classifier")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--output", help="Path to save evaluation results")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluation pipeline
        pipeline = EvaluationPipeline(config_path=args.config)
        
        # Run evaluation
        results = pipeline.evaluate_model(
            model_path=args.model,
            test_data_path=args.test_data,
            output_path=args.output,
            generate_plots=args.plots
        )
        
        # Print results
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
        logger.info(f"Test AUC: {results['metrics']['roc_auc']:.4f}")
        logger.info(f"Test F1: {results['metrics']['f1_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
