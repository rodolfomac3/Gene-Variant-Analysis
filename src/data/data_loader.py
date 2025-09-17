"""
Data loading utilities for gene variant analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging
import yaml

from config.logging_config import get_logger

logger = get_logger(__name__)


class GenomicDataLoader:
    """
    Data loader class for handling various genomic data formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GenomicDataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})
        
    def load_vcf(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load VCF (Variant Call Format) file.
        
        Args:
            file_path: Path to the VCF file
            
        Returns:
            DataFrame containing variant data
        """
        logger.info(f"Loading VCF file: {file_path}")
        
        try:
            # Read VCF file using pandas
            vcf_data = pd.read_csv(
                file_path,
                sep='\t',
                comment='#',
                low_memory=False
            )
            
            # Standardize column names
            vcf_data.columns = vcf_data.columns.str.upper()
            
            logger.info(f"Successfully loaded VCF file with {len(vcf_data)} variants")
            return vcf_data
            
        except Exception as e:
            logger.error(f"Error loading VCF file: {e}")
            raise
    
    def load_annotations(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load variant annotations.
        
        Args:
            file_path: Path to the annotation file
            
        Returns:
            DataFrame containing annotation data
        """
        logger.info(f"Loading annotations: {file_path}")
        
        try:
            if str(file_path).endswith('.csv'):
                data = pd.read_csv(file_path)
            elif str(file_path).endswith('.tsv'):
                data = pd.read_csv(file_path, sep='\t')
            elif str(file_path).endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                data = pd.read_csv(file_path)
            
            logger.info(f"Successfully loaded annotations with {len(data)} entries")
            return data
            
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            raise
    
    def merge_data_sources(self, variants_df: pd.DataFrame, 
                          annotations_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge variant data with annotations.
        
        Args:
            variants_df: Variant data
            annotations_df: Optional annotation data
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging data sources")
        
        merged_df = variants_df.copy()
        
        if annotations_df is not None:
            # Try to merge on common columns
            merge_cols = ['CHROM', 'POS', 'REF', 'ALT']
            available_cols = [col for col in merge_cols if col in variants_df.columns and col in annotations_df.columns]
            
            if available_cols:
                merged_df = merged_df.merge(
                    annotations_df,
                    on=available_cols,
                    how='left'
                )
                logger.info(f"Merged on columns: {available_cols}")
            else:
                logger.warning("No common columns found for merging, concatenating instead")
                merged_df = pd.concat([variants_df, annotations_df], axis=1)
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        return merged_df
    
    def split_data(self, data: pd.DataFrame, target_col: str,
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data
            target_col: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random state
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("Splitting data into train/validation/test sets")
        
        # First split: train+val vs test
        train_val, test_df = train_test_split(
            data, test_size=test_size, random_state=random_state, stratify=data[target_col]
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state, stratify=train_val[target_col]
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_data(self, data: pd.DataFrame, file_path: Union[str, Path], 
                  file_type: str = 'csv', **kwargs) -> None:
        """
        Save DataFrame to file.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            file_type: Type of file to save
            **kwargs: Additional arguments for the save function
        """
        logger.info(f"Saving data to {file_type} file: {file_path}")
        
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            if file_type == 'csv':
                data.to_csv(file_path, index=False, **kwargs)
            elif file_type == 'parquet':
                data.to_parquet(file_path, index=False, **kwargs)
            elif file_type == 'json':
                data.to_json(file_path, orient='records', **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            logger.info(f"Successfully saved data to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise