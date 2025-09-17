"""
Tests for data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data.data_loader import GenomicDataLoader


class TestGenomicDataLoader:
    """Test cases for GenomicDataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'data': {
                'raw': 'data/raw',
                'processed': 'data/processed',
                'external': 'data/external',
                'interim': 'data/interim'
            }
        }
        self.loader = GenomicDataLoader(self.config)
    
    def test_init(self):
        """Test GenomicDataLoader initialization."""
        assert self.loader.config == self.config
        assert self.loader.data_config == self.config['data']
    
    def test_load_csv(self):
        """Test loading CSV files."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('col1,col2,col3\n1,2,3\n4,5,6\n')
            temp_path = f.name
        
        try:
            data = self.loader.load_csv(temp_path)
            assert isinstance(data, pd.DataFrame)
            assert data.shape == (2, 3)
            assert list(data.columns) == ['col1', 'col2', 'col3']
        finally:
            os.unlink(temp_path)
    
    def test_load_parquet(self):
        """Test loading Parquet files."""
        # Create temporary Parquet file
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            test_data.to_parquet(f.name)
            temp_path = f.name
        
        try:
            data = self.loader.load_parquet(temp_path)
            assert isinstance(data, pd.DataFrame)
            assert data.shape == (3, 2)
            assert list(data.columns) == ['col1', 'col2']
        finally:
            os.unlink(temp_path)
    
    def test_merge_data_sources(self):
        """Test merging data sources."""
        # Create test dataframes
        variants_df = pd.DataFrame({
            'CHROM': ['chr1', 'chr2'],
            'POS': [100, 200],
            'REF': ['A', 'T'],
            'ALT': ['G', 'C']
        })
        
        annotations_df = pd.DataFrame({
            'CHROM': ['chr1', 'chr2'],
            'POS': [100, 200],
            'gene_name': ['GENE1', 'GENE2'],
            'consequence': ['missense', 'synonymous']
        })
        
        merged_df = self.loader.merge_data_sources(variants_df, annotations_df)
        
        assert isinstance(merged_df, pd.DataFrame)
        assert len(merged_df) == 2
        assert 'gene_name' in merged_df.columns
        assert 'consequence' in merged_df.columns
    
    def test_split_data(self):
        """Test data splitting."""
        # Create test data
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        train_df, val_df, test_df = self.loader.split_data(
            data, target_col='target', test_size=0.2, val_size=0.1
        )
        
        assert len(train_df) + len(val_df) + len(test_df) == 100
        assert len(test_df) == 20  # 20% of 100
        assert len(val_df) == 10   # 10% of 100
        assert len(train_df) == 70  # remaining
    
    def test_save_data(self):
        """Test saving data."""
        # Create test data
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            self.loader.save_data(data, temp_path, file_type='csv')
            
            # Verify file was created and contains correct data
            loaded_data = pd.read_csv(temp_path)
            pd.testing.assert_frame_equal(data, loaded_data)
        finally:
            os.unlink(temp_path)
