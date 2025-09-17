### src/models/ensemble_model.py
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from loguru import logger
import mlflow
from .variant_classifier import VariantClassifier

class EnsembleVariantClassifier:
    """Ensemble model combining multiple classifiers"""
    
    def __init__(self, config: Dict[str, Any], ensemble_type: str = "voting"):
        self.config = config
        self.ensemble_type = ensemble_type
        self.base_models = {}
        self.ensemble_model = None
        self.feature_importance = None
        
    def build_ensemble(
        self,
        model_types: List[str] = ["xgboost", "lightgbm", "random_forest"]
    ):
        """
        Build ensemble model
        
        Args:
            model_types: List of base model types
        """
        logger.info(f"Building {self.ensemble_type} ensemble with models: {model_types}")
        
        # Initialize base models
        for model_type in model_types:
            self.base_models[model_type] = VariantClassifier(
                self.config,
                model_type=model_type
            )
        
        # Create ensemble
        if self.ensemble_type == "voting":
            self.ensemble_model = VotingClassifier(
                estimators=[(name, model.model) for name, model in self.base_models.items()],
                voting='soft'
            )
        elif self.ensemble_type == "stacking":
            self.ensemble_model = StackingClassifier(
                estimators=[(name, model.model) for name, model in self.base_models.items()],
                final_estimator=LogisticRegression(),
                cv=5
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        track_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            track_mlflow: Whether to track with MLflow
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.ensemble_type} ensemble")
        
        if track_mlflow:
            mlflow.start_run(nested=True)
            mlflow.log_param("ensemble_type", self.ensemble_type)
            mlflow.log_param("base_models", list(self.base_models.keys()))
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate
        metrics = {}
        
        # Training metrics
        y_train_pred = self.ensemble_model.predict(X_train)
        y_train_proba = self.ensemble_model.predict_proba(X_train)[:, 1]
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train_auc'] = roc_auc_score(y_train, y_train_proba)
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.ensemble_model.predict(X_val)
            y_val_proba = self.ensemble_model.predict_proba(X_val)[:, 1]
            
            metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
            metrics['val_auc'] = roc_auc_score(y_val, y_val_proba)
        
        if track_mlflow:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.sklearn.log_model(self.ensemble_model, "ensemble_model")
            mlflow.end_run()
        
        logger.info(f"Ensemble training completed. Val AUC: {metrics.get('val_auc', 'N/A'):.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        return self.ensemble_model.predict_proba(X)