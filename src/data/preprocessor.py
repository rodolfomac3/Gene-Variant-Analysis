"""
Data preprocessing utilities for gene variant analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)


class GenomicPreprocessor:
    """
    Data preprocessing class for gene variant analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GenomicPreprocessor."""
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit preprocessing transformers on training data.
        
        Args:
            data: Training data
        """
        logger.info("Fitting preprocessing transformers")
        
        # Identify column types
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        # Fit imputers
        for col in numeric_columns:
            if data[col].isnull().any():
                self.imputers[f'{col}_imputer'] = SimpleImputer(strategy='median')
                self.imputers[f'{col}_imputer'].fit(data[[col]])
        
        for col in categorical_columns:
            if data[col].isnull().any():
                self.imputers[f'{col}_imputer'] = SimpleImputer(strategy='most_frequent')
                self.imputers[f'{col}_imputer'].fit(data[[col]])
        
        # Fit scalers for numeric columns
        for col in numeric_columns:
            self.scalers[col] = StandardScaler()
            self.scalers[col].fit(data[[col]])
        
        # Fit encoders for categorical columns
        for col in categorical_columns:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(data[col].astype(str))
        
        self.is_fitted = True
        logger.info("Preprocessing transformers fitted")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming data")
        
        transformed_data = data.copy()
        
        # Apply imputation
        for col in data.columns:
            imputer_key = f'{col}_imputer'
            if imputer_key in self.imputers:
                if data[col].dtype in ['object']:
                    transformed_data[col] = self.imputers[imputer_key].transform(data[[col]]).flatten()
                else:
                    transformed_data[col] = self.imputers[imputer_key].transform(data[[col]]).flatten()
        
        # Apply scaling for numeric columns
        for col in data.select_dtypes(include=[np.number]).columns:
            if col in self.scalers:
                transformed_data[col] = self.scalers[col].transform(data[[col]]).flatten()
        
        # Apply encoding for categorical columns
        for col in data.select_dtypes(include=['object']).columns:
            if col in self.encoders:
                transformed_data[col] = self.encoders[col].transform(data[col].astype(str))
        
        logger.info("Data transformation completed")
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input data
            
        Returns:
            Data with handled missing values
        """
        logger.info("Handling missing values")
        
        cleaned_data = data.copy()
        
        # Handle numeric columns
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if cleaned_data[col].isnull().any():
                median_value = cleaned_data[col].median()
                cleaned_data[col].fillna(median_value, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        # Handle categorical columns
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if cleaned_data[col].isnull().any():
                mode_value = cleaned_data[col].mode()[0] if not cleaned_data[col].mode().empty else 'Unknown'
                cleaned_data[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        logger.info("Missing value handling completed")
        return cleaned_data
    
    def remove_outliers(self, data: pd.DataFrame, 
                       numeric_columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numeric columns.
        
        Args:
            data: Input data
            numeric_columns: List of numeric columns to process
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Data with outliers removed
        """
        logger.info(f"Removing outliers using {method} method")
        
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        cleaned_data = data.copy()
        initial_rows = len(cleaned_data)
        
        for col in numeric_columns:
            if col not in data.columns:
                continue
            
            if method == 'iqr':
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                outliers = z_scores > threshold
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            cleaned_data = cleaned_data[~outliers]
            logger.info(f"Removed {outliers.sum()} outliers from {col}")
        
        final_rows = len(cleaned_data)
        logger.info(f"Outlier removal completed. Removed {initial_rows - final_rows} rows")
        
        return cleaned_data
    
    def save(self, file_path: str) -> None:
        """
        Save preprocessor to file.
        
        Args:
            file_path: Path to save the preprocessor
        """
        preprocessor_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        
        joblib.dump(preprocessor_data, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'GenomicPreprocessor':
        """
        Load preprocessor from file.
        
        Args:
            file_path: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        preprocessor_data = joblib.load(file_path)
        
        preprocessor = cls(preprocessor_data['config'])
        preprocessor.scalers = preprocessor_data['scalers']
        preprocessor.encoders = preprocessor_data['encoders']
        preprocessor.imputers = preprocessor_data['imputers']
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor