#!/usr/bin/env python
"""
Prediction script for genomic variant analysis pipeline
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.inference_pipeline import InferencePipeline
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Make predictions on genomic variants")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to input data")
    parser.add_argument("--output", help="Path to save predictions")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--probabilities", action="store_true", help="Output prediction probabilities")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference pipeline
        pipeline = InferencePipeline(config_path=args.config)
        
        # Make predictions
        predictions = pipeline.predict(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            return_probabilities=args.probabilities
        )
        
        # Print results
        logger.info(f"Predictions completed successfully!")
        logger.info(f"Predicted {len(predictions)} variants")
        
        if args.probabilities:
            logger.info(f"Prediction probabilities saved to {args.output}")
        else:
            logger.info(f"Predictions saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
