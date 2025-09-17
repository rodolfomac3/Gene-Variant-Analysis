"""
Variant classifier implementation for genomic variant analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)


class VariantClassifier:
    """Multi-model variant classifier with MLflow tracking"""
    
    def __init__(self, config: Dict[str, Any], model_type: str = "xgboost"):
        self.config = config
        self.model_config = config['model']
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the specified model"""
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                **self.model_config.get('xgboost', {}),
                random_state=self.model_config.get('random_state', 42)
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                **self.model_config.get('lightgbm', {}),
                random_state=self.model_config.get('random_state', 42)
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.model_config.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.model_config.get('random_state', 42)
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=self.model_config.get('random_state', 42),
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        track_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with optional MLflow tracking
        """
        logger.info(f"Training {self.model_type} model")
        
        if track_mlflow:
            mlflow.start_run(nested=True)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_params(self.model_config.get(self.model_type, {}))
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Evaluate on training data
        train_metrics = self.evaluate(X_train, y_train, prefix="train")
        
        # Evaluate on validation data if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, prefix="val")
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        if track_mlflow:
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            if self.model_type == "xgboost":
                mlflow.xgboost.log_model(self.model, "model")
            else:
                mlflow.sklearn.log_model(self.model, "model")
            
            # Log feature importance
            if self.feature_importance is not None:
                mlflow.log_text(
                    self.feature_importance.to_csv(index=False),
                    "feature_importance.csv"
                )
            
            mlflow.end_run()
        
        logger.info(f"Training completed. Val AUC: {val_metrics.get('val_auc', 'N/A')}")
        
        return metrics
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None
        
        metrics = {
            f"{prefix}_accuracy": accuracy_score(y, y_pred),
            f"{prefix}_precision": precision_score(y, y_pred, average='weighted'),
            f"{prefix}_recall": recall_score(y, y_pred, average='weighted'),
            f"{prefix}_f1": f1_score(y, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics[f"{prefix}_auc"] = roc_auc_score(y, y_pred_proba)
            metrics[f"{prefix}_avg_precision"] = average_precision_score(y, y_pred_proba)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save model to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'VariantClassifier':
        """Load model from file"""
        model_data = joblib.load(path)
        
        classifier = cls(model_data['config'], model_data['model_type'])
        classifier.model = model_data['model']
        classifier.feature_importance = model_data['feature_importance']
        
        logger.info(f"Loaded model from {path}")
        return classifier