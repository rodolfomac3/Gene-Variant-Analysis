"""
Tests for model classes.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.models.variant_classifier import VariantClassifier
from src.models.ensemble_model import EnsembleVariantClassifier


class TestVariantClassifier:
    """Test cases for VariantClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                },
                'random_state': 42
            }
        }
        self.classifier = VariantClassifier(self.config, model_type='xgboost')
    
    def test_init(self):
        """Test VariantClassifier initialization."""
        assert self.classifier.config == self.config
        assert self.classifier.model_type == 'xgboost'
        assert self.classifier.model is not None
    
    def test_predict(self):
        """Test prediction method."""
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y_test = np.array([0, 1, 0, 1, 0])
        
        # Train model
        self.classifier.model.fit(X_test, y_test)
        
        # Make predictions
        predictions = self.classifier.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction method."""
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y_test = np.array([0, 1, 0, 1, 0])
        
        # Train model
        self.classifier.model.fit(X_test, y_test)
        
        # Make probability predictions
        probabilities = self.classifier.predict_proba(X_test)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_evaluate(self):
        """Test evaluation method."""
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y_test = np.array([0, 1, 0, 1, 0])
        
        # Train model
        self.classifier.model.fit(X_test, y_test)
        
        # Evaluate model
        metrics = self.classifier.evaluate(X_test, y_test, prefix="test")
        
        assert isinstance(metrics, dict)
        assert 'test_accuracy' in metrics
        assert 'test_precision' in metrics
        assert 'test_recall' in metrics
        assert 'test_f1' in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, (int, float)))


class TestEnsembleVariantClassifier:
    """Test cases for EnsembleVariantClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                },
                'random_state': 42
            }
        }
        self.ensemble = EnsembleVariantClassifier(self.config, ensemble_type='voting')
    
    def test_init(self):
        """Test EnsembleVariantClassifier initialization."""
        assert self.ensemble.config == self.config
        assert self.ensemble.ensemble_type == 'voting'
        assert self.ensemble.base_models == {}
        assert self.ensemble.ensemble_model is None
    
    def test_build_ensemble(self):
        """Test ensemble building."""
        model_types = ['xgboost', 'lightgbm']
        self.ensemble.build_ensemble(model_types)
        
        assert len(self.ensemble.base_models) == 2
        assert 'xgboost' in self.ensemble.base_models
        assert 'lightgbm' in self.ensemble.base_models
        assert self.ensemble.ensemble_model is not None
    
    def test_predict(self):
        """Test ensemble prediction."""
        # Build ensemble
        model_types = ['xgboost', 'lightgbm']
        self.ensemble.build_ensemble(model_types)
        
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y_test = np.array([0, 1, 0, 1, 0])
        
        # Train ensemble
        self.ensemble.ensemble_model.fit(X_test, y_test)
        
        # Make predictions
        predictions = self.ensemble.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test ensemble probability prediction."""
        # Build ensemble
        model_types = ['xgboost', 'lightgbm']
        self.ensemble.build_ensemble(model_types)
        
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        y_test = np.array([0, 1, 0, 1, 0])
        
        # Train ensemble
        self.ensemble.ensemble_model.fit(X_test, y_test)
        
        # Make probability predictions
        probabilities = self.ensemble.predict_proba(X_test)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
