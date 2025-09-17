"""
Tests for pipeline modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.pipeline.inference_pipeline import InferencePipeline


class TestTrainingPipeline:
    """Test cases for TrainingPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'data': {
                'raw': 'data/raw',
                'processed': 'data/processed',
                'external': 'data/external',
                'interim': 'data/interim'
            },
            'model': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'random_state': 42,
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'mlflow': {
                'tracking_uri': 'file:./mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(self.config, f)
            self.config_path = f.name
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'config_path'):
            os.unlink(self.config_path)
    
    @patch('src.pipeline.training_pipeline.TrainingPipeline.__init__')
    def test_init(self, mock_init):
        """Test TrainingPipeline initialization."""
        mock_init.return_value = None
        
        pipeline = TrainingPipeline(self.config_path)
        mock_init.assert_called_once_with(self.config_path)
    
    @patch('src.pipeline.training_pipeline.mlflow.start_run')
    @patch('src.pipeline.training_pipeline.GenomicDataLoader')
    @patch('src.pipeline.training_pipeline.VariantFeatureExtractor')
    @patch('src.pipeline.training_pipeline.GenomicDataValidator')
    @patch('src.pipeline.training_pipeline.GenomicPreprocessor')
    @patch('src.pipeline.training_pipeline.VariantClassifier')
    def test_run(self, mock_classifier, mock_preprocessor, mock_validator, 
                 mock_feature_extractor, mock_data_loader, mock_mlflow):
        """Test training pipeline run method."""
        # Mock the pipeline initialization
        pipeline = TrainingPipeline(self.config_path)
        pipeline.config = self.config
        pipeline.data_loader = mock_data_loader.return_value
        pipeline.preprocessor = mock_preprocessor.return_value
        pipeline.validator = mock_validator.return_value
        pipeline.feature_extractor = mock_feature_extractor.return_value
        
        # Mock data loading
        mock_variants_df = pd.DataFrame({
            'CHROM': ['chr1', 'chr2'],
            'POS': [100, 200],
            'REF': ['A', 'T'],
            'ALT': ['G', 'C']
        })
        mock_data_loader.return_value.load_vcf.return_value = mock_variants_df
        mock_data_loader.return_value.merge_data_sources.return_value = mock_variants_df
        mock_data_loader.return_value.split_data.return_value = (
            pd.DataFrame({'feature1': [1, 2], 'pathogenicity': [0, 1]}),
            pd.DataFrame({'feature1': [3], 'pathogenicity': [0]}),
            pd.DataFrame({'feature1': [4], 'pathogenicity': [1]})
        )
        
        # Mock validation
        mock_validator.return_value.validate_variants.return_value = (True, [])
        mock_validator.return_value.generate_data_quality_report.return_value = {}
        
        # Mock feature extraction
        mock_features_df = pd.DataFrame({
            'feature1': [1, 2],
            'pathogenicity': [0, 1]
        })
        mock_feature_extractor.return_value.extract_basic_features.return_value = mock_features_df
        mock_feature_extractor.return_value.extract_conservation_features.return_value = mock_features_df
        mock_feature_extractor.return_value.extract_population_features.return_value = mock_features_df
        
        # Mock preprocessing
        mock_preprocessor.return_value.handle_missing_values.return_value = mock_features_df
        mock_preprocessor.return_value.fit.return_value = None
        mock_preprocessor.return_value.transform.return_value = mock_features_df
        mock_preprocessor.return_value.save.return_value = None
        
        # Mock model training
        mock_classifier.return_value.train.return_value = {'val_auc': 0.8}
        mock_classifier.return_value.evaluate.return_value = {'test_auc': 0.75}
        mock_classifier.return_value.save.return_value = None
        
        # Mock MLflow
        mock_mlflow.return_value.__enter__ = Mock()
        mock_mlflow.return_value.__exit__ = Mock()
        
        # Create temporary VCF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
            f.write('chr1\t100\t.\tA\tG\t30\tPASS\t.\n')
            f.write('chr2\t200\t.\tT\tC\t25\tPASS\t.\n')
            vcf_path = f.name
        
        try:
            # Run pipeline
            results = pipeline.run(vcf_path, model_type='xgboost')
            
            # Verify results
            assert isinstance(results, dict)
            assert 'run_id' in results
            assert 'status' in results
            assert results['status'] == 'completed'
        finally:
            os.unlink(vcf_path)


class TestEvaluationPipeline:
    """Test cases for EvaluationPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'mlflow': {
                'tracking_uri': 'file:./mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(self.config, f)
            self.config_path = f.name
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'config_path'):
            os.unlink(self.config_path)
    
    @patch('src.pipeline.evaluation_pipeline.EvaluationPipeline.__init__')
    def test_init(self, mock_init):
        """Test EvaluationPipeline initialization."""
        mock_init.return_value = None
        
        pipeline = EvaluationPipeline(self.config_path)
        mock_init.assert_called_once_with(self.config_path)


class TestInferencePipeline:
    """Test cases for InferencePipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'mlflow': {
                'tracking_uri': 'file:./mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(self.config, f)
            self.config_path = f.name
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'config_path'):
            os.unlink(self.config_path)
    
    @patch('src.pipeline.inference_pipeline.InferencePipeline.__init__')
    def test_init(self, mock_init):
        """Test InferencePipeline initialization."""
        mock_init.return_value = None
        
        pipeline = InferencePipeline(self.config_path)
        mock_init.assert_called_once_with(self.config_path)
