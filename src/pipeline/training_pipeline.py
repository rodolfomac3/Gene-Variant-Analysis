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
from ..data.validator import GenomicDataValidator
from ..features.variant_features import VariantFeatureExtractor
from ..models.variant_classifier import VariantClassifier
from ..models.ensemble_model import EnsembleVariantClassifier
from ..utils.metrics import calculate_metrics, plot_metrics

class TrainingPipeline:
    """End-to-end training pipeline with MLflow tracking"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = GenomicDataLoader(self.config)
        self.preprocessor = GenomicPreprocessor(self.config)
        self.validator = GenomicDataValidator(self.config)
        self.feature_extractor = VariantFeatureExtractor(self.config)
        
        # MLflow setup
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
    def run(
        self,
        vcf_path: str,
        annotation_path: Optional[str] = None,
        model_type: str = "xgboost",
        use_ensemble: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            vcf_path: Path to VCF file
            annotation_path: Optional path to annotation file
            model_type: Type of model to train
            use_ensemble: Whether to use ensemble model
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting training pipeline")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with mlflow.start_run(run_name=f"training_pipeline_{run_id}"):
            # Log parameters
            mlflow.log_param("vcf_path", vcf_path)
            mlflow.log_param("annotation_path", annotation_path or "None")
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("use_ensemble", use_ensemble)
            
            # Step 1: Load Data
            logger.info("Step 1: Loading data")
            variants_df = self.data_loader.load_vcf(vcf_path)
            
            annotations_df = None
            if annotation_path:
                annotations_df = self.data_loader.load_annotations(annotation_path)
            
            # Merge data sources
            data_df = self.data_loader.merge_data_sources(
                variants_df,
                annotations_df
            )
            
            mlflow.log_metric("n_variants", len(data_df))
            
            # Step 2: Validate Data
            logger.info("Step 2: Validating data")
            is_valid, errors = self.validator.validate_variants(variants_df)
            if not is_valid:
                logger.error(f"Data validation failed: {errors}")
                mlflow.log_param("validation_status", "failed")
                mlflow.log_text("\n".join(errors), "validation_errors.txt")
                return {"status": "failed", "errors": errors}
            
            mlflow.log_param("validation_status", "passed")
            
            # Generate data quality report
            quality_report = self.validator.generate_data_quality_report(data_df)
            mlflow.log_dict(quality_report, "data_quality_report.json")
            
            # Step 3: Feature Engineering
            logger.info("Step 3: Feature engineering")
            features_df = self.feature_extractor.extract_basic_features(data_df)
            
            # Add conservation features if available
            features_df = self.feature_extractor.extract_conservation_features(features_df)
            
            # Add population features if available
            features_df = self.feature_extractor.extract_population_features(features_df)
            
            mlflow.log_metric("n_features", len(features_df.columns))
            
            # Step 4: Preprocessing
            logger.info("Step 4: Preprocessing data")
            
            # Handle missing values
            features_df = self.preprocessor.handle_missing_values(features_df)
            
            # Split data
            # Assuming we have a target column (you'll need to adjust this)
            target_col = 'pathogenicity'  # Example target
            
            if target_col not in features_df.columns:
                # Create synthetic target for demonstration
                logger.warning(f"Target column '{target_col}' not found. Creating synthetic target.")
                features_df[target_col] = np.random.randint(0, 2, len(features_df))
            
            train_df, val_df, test_df = self.data_loader.split_data(
                features_df,
                target_col=target_col,
                test_size=self.config['model']['test_size'],
                val_size=self.config['model']['validation_size']
            )
            
            # Separate features and target
            feature_cols = [col for col in train_df.columns if col != target_col]
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            # Fit and transform preprocessing
            self.preprocessor.fit(X_train)
            X_train = self.preprocessor.transform(X_train)
            X_val = self.preprocessor.transform(X_val)
            X_test = self.preprocessor.transform(X_test)
            
            # Save preprocessor
            preprocessor_path = f"models/artifacts/preprocessor_{run_id}.pkl"
            self.preprocessor.save(preprocessor_path)
            mlflow.log_artifact(preprocessor_path)
            
            # Step 5: Model Training
            logger.info("Step 5: Training model")
            
            if use_ensemble:
                model = EnsembleVariantClassifier(self.config)
                model.build_ensemble()
                metrics = model.train(X_train, y_train, X_val, y_val, track_mlflow=True)
            else:
                model = VariantClassifier(self.config, model_type=model_type)
                metrics = model.train(X_train, y_train, X_val, y_val, track_mlflow=True)
            
            # Step 6: Evaluation on Test Set
            logger.info("Step 6: Evaluating on test set")
            test_metrics = model.evaluate(X_test, y_test, prefix="test")
            
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Save model
            model_path = f"models/trained/model_{run_id}.pkl"
            model.save(model_path)
            mlflow.log_artifact(model_path)
            
            # Log feature importance
            if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                feature_importance_path = f"models/artifacts/feature_importance_{run_id}.csv"
                model.feature_importance.to_csv(feature_importance_path, index=False)
                mlflow.log_artifact(feature_importance_path)
            
            # Step 7: Generate Reports
            logger.info("Step 7: Generating reports")
            
            # Create comprehensive results
            results = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "data_stats": {
                    "n_variants": len(data_df),
                    "n_features": len(feature_cols),
                    "n_train": len(train_df),
                    "n_val": len(val_df),
                    "n_test": len(test_df)
                },
                "metrics": {**metrics, **test_metrics},
                "model_type": model_type if not use_ensemble else "ensemble",
                "status": "completed"
            }
            
            # Save results
            results_path = f"models/artifacts/results_{run_id}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            mlflow.log_artifact(results_path)
            
            logger.info("Training pipeline completed successfully")
            
            return results
    
    def run_hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str = "xgboost",
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning using Optuna
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Type of model to tune
            n_trials: Number of Optuna trials
            
        Returns:
            Best parameters and metrics
        """
        import optuna
        from optuna.integration import MLflowCallback
        
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        def objective(trial):
            # Suggest hyperparameters
            if model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                }
            elif model_type == "lightgbm":
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
                }
            else:
                raise ValueError(f"Hyperparameter tuning not implemented for {model_type}")
            
            # Update config with suggested parameters
            config = self.config.copy()
            config['model'][model_type].update(params)
            
            # Train model
            model = VariantClassifier(config, model_type=model_type)
            metrics = model.train(X_train, y_train, X_val, y_val, track_mlflow=False)
            
            return metrics['val_auc']
        
        # Create study
        study = optuna.create_study(
            study_name=f"{model_type}_optimization",
            direction='maximize'
        )
        
        # Add MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.config['mlflow']['tracking_uri'],
            metric_name="val_auc"
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[mlflow_callback]
        )
        
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_params}")
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials)
        }