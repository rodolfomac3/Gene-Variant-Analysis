"""
Data validation utilities for gene variant analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)


class GenomicDataValidator:
    """
    Data validation class for gene variant analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GenomicDataValidator."""
        self.config = config
        self.required_vcf_columns = [
            'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'
        ]
        
    def validate_variants(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate variant data.
        
        Args:
            data: Variant data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        logger.info("Validating variant data")
        
        errors = []
        
        # Check required columns
        missing_columns = set(self.required_vcf_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Validate chromosome data
        if 'CHROM' in data.columns:
            if not self._validate_chromosomes(data['CHROM']):
                errors.append("Invalid chromosome values found")
        
        # Validate position data
        if 'POS' in data.columns:
            if not self._validate_positions(data['POS']):
                errors.append("Invalid position values found")
        
        # Validate allele data
        if 'REF' in data.columns and 'ALT' in data.columns:
            if not self._validate_alleles(data['REF'], data['ALT']):
                errors.append("Invalid allele data found")
        
        # Check for duplicates
        if data.duplicated().any():
            errors.append("Duplicate rows found")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Variant data validation passed")
        else:
            logger.error(f"Variant data validation failed: {errors}")
        
        return is_valid, errors
    
    def generate_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        logger.info("Generating data quality report")
        
        report = {
            'basic_stats': {
                'n_rows': len(data),
                'n_columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
            },
            'missing_data': {
                'total_missing': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
                'columns_with_missing': data.isnull().sum()[data.isnull().sum() > 0].to_dict()
            },
            'duplicate_data': {
                'n_duplicates': data.duplicated().sum(),
                'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100
            },
            'data_types': data.dtypes.to_dict(),
            'numeric_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }
        
        # Categorical column summaries
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            report['categorical_summary'][col] = {
                'unique_values': data[col].nunique(),
                'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                'frequency': data[col].value_counts().head().to_dict()
            }
        
        logger.info("Data quality report generated")
        return report
    
    def _validate_chromosomes(self, chrom_series: pd.Series) -> bool:
        """Validate chromosome values."""
        valid_chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
        valid_chromosomes.extend([str(i) for i in range(1, 23)] + ['X', 'Y', 'M'])
        
        invalid_chromosomes = set(chrom_series.unique()) - set(valid_chromosomes)
        
        if invalid_chromosomes:
            logger.warning(f"Found invalid chromosome values: {invalid_chromosomes}")
            return False
        
        return True
    
    def _validate_positions(self, pos_series: pd.Series) -> bool:
        """Validate position values."""
        if not pd.api.types.is_numeric_dtype(pos_series):
            logger.error("Position column is not numeric")
            return False
        
        if (pos_series <= 0).any():
            logger.error("Found non-positive positions")
            return False
        
        return True
    
    def _validate_alleles(self, ref_series: pd.Series, alt_series: pd.Series) -> bool:
        """Validate reference and alternate alleles."""
        valid_nucleotides = set('ATCGN')
        
        ref_invalid = set(ref_series.str.upper().str.cat()) - valid_nucleotides
        alt_invalid = set(alt_series.str.upper().str.cat()) - valid_nucleotides
        
        if ref_invalid:
            logger.warning(f"Found invalid reference alleles: {ref_invalid}")
        if alt_invalid:
            logger.warning(f"Found invalid alternate alleles: {alt_invalid}")
        
        if ref_series.isnull().any() or alt_series.isnull().any():
            logger.error("Found null values in allele columns")
            return False
        
        return True
