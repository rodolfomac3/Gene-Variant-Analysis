"""
VCF (Variant Call Format) parsing utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)


class VCFParser:
    """
    VCF file parser for genomic variant analysis.
    """
    
    def __init__(self):
        """Initialize VCFParser."""
        self.header_lines = []
        self.info_fields = {}
        self.format_fields = {}
        
    def parse_vcf(self, file_path: str) -> pd.DataFrame:
        """
        Parse VCF file and return DataFrame.
        
        Args:
            file_path: Path to VCF file
            
        Returns:
            DataFrame with parsed VCF data
        """
        logger.info(f"Parsing VCF file: {file_path}")
        
        try:
            # Read VCF file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Extract header and data
            header_lines, data_lines = self._separate_header_data(lines)
            self.header_lines = header_lines
            
            # Parse header to extract INFO and FORMAT fields
            self._parse_header(header_lines)
            
            # Parse data lines
            variants_df = self._parse_data_lines(data_lines)
            
            logger.info(f"Successfully parsed VCF with {len(variants_df)} variants")
            return variants_df
            
        except Exception as e:
            logger.error(f"Error parsing VCF file: {e}")
            raise
    
    def _separate_header_data(self, lines: List[str]) -> Tuple[List[str], List[str]]:
        """Separate header lines from data lines."""
        header_lines = []
        data_lines = []
        
        for line in lines:
            if line.startswith('#'):
                header_lines.append(line.strip())
            else:
                data_lines.append(line.strip())
        
        return header_lines, data_lines
    
    def _parse_header(self, header_lines: List[str]) -> None:
        """Parse VCF header to extract INFO and FORMAT fields."""
        for line in header_lines:
            if line.startswith('##INFO='):
                info_field = self._parse_info_format_line(line)
                self.info_fields[info_field['ID']] = info_field
            elif line.startswith('##FORMAT='):
                format_field = self._parse_info_format_line(line)
                self.format_fields[format_field['ID']] = format_field
    
    def _parse_info_format_line(self, line: str) -> Dict[str, str]:
        """Parse INFO or FORMAT header line."""
        # Remove ##INFO= or ##FORMAT=
        content = line.split('=', 1)[1]
        
        # Parse key-value pairs
        field_info = {}
        parts = content.split(',')
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                field_info[key] = value.strip('"')
        
        return field_info
    
    def _parse_data_lines(self, data_lines: List[str]) -> pd.DataFrame:
        """Parse VCF data lines into DataFrame."""
        if not data_lines:
            return pd.DataFrame()
        
        # Split first line to get column names
        first_line = data_lines[0].split('\t')
        n_columns = len(first_line)
        
        # Parse all data lines
        parsed_data = []
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == n_columns:
                parsed_data.append(parts)
            else:
                logger.warning(f"Skipping malformed line: {line[:100]}...")
        
        # Create DataFrame
        if not parsed_data:
            return pd.DataFrame()
        
        # Get column names from header
        column_names = self._get_column_names()
        
        # Ensure we have the right number of columns
        if len(column_names) != n_columns:
            # Use standard VCF columns
            column_names = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
            if n_columns > 8:
                column_names.extend([f'FORMAT_{i}' for i in range(n_columns - 8)])
        
        df = pd.DataFrame(parsed_data, columns=column_names)
        
        # Convert numeric columns
        numeric_columns = ['POS', 'QUAL']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _get_column_names(self) -> List[str]:
        """Get column names from header."""
        for line in self.header_lines:
            if line.startswith('#CHROM'):
                return line[1:].split('\t')
        
        # Default VCF columns
        return ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
    
    def parse_info_field(self, info_string: str) -> Dict[str, Any]:
        """
        Parse INFO field string into dictionary.
        
        Args:
            info_string: INFO field string
            
        Returns:
            Dictionary with parsed INFO fields
        """
        info_dict = {}
        
        if pd.isna(info_string) or info_string == '.':
            return info_dict
        
        # Split by semicolon
        fields = info_string.split(';')
        
        for field in fields:
            if '=' in field:
                key, value = field.split('=', 1)
                
                # Try to convert to appropriate type
                if value.isdigit():
                    info_dict[key] = int(value)
                elif value.replace('.', '').isdigit():
                    info_dict[key] = float(value)
                elif value.upper() in ['TRUE', 'FALSE']:
                    info_dict[key] = value.upper() == 'TRUE'
                else:
                    info_dict[key] = value
            else:
                # Flag field (no value)
                info_dict[field] = True
        
        return info_dict
    
    def parse_format_field(self, format_string: str, sample_string: str) -> Dict[str, Any]:
        """
        Parse FORMAT and sample data.
        
        Args:
            format_string: FORMAT field string
            sample_string: Sample data string
            
        Returns:
            Dictionary with parsed sample data
        """
        if pd.isna(format_string) or pd.isna(sample_string):
            return {}
        
        format_fields = format_string.split(':')
        sample_values = sample_string.split(':')
        
        sample_dict = {}
        for field, value in zip(format_fields, sample_values):
            # Try to convert to appropriate type
            if value.isdigit():
                sample_dict[field] = int(value)
            elif value.replace('.', '').isdigit():
                sample_dict[field] = float(value)
            else:
                sample_dict[field] = value
        
        return sample_dict
    
    def extract_info_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract INFO field columns into separate columns.
        
        Args:
            df: DataFrame with INFO column
            
        Returns:
            DataFrame with extracted INFO columns
        """
        logger.info("Extracting INFO field columns")
        
        if 'INFO' not in df.columns:
            logger.warning("No INFO column found")
            return df
        
        # Parse INFO field for each row
        info_data = []
        for info_string in df['INFO']:
            info_dict = self.parse_info_field(info_string)
            info_data.append(info_dict)
        
        # Create DataFrame from INFO data
        info_df = pd.DataFrame(info_data)
        
        # Add prefix to avoid column name conflicts
        info_df.columns = [f'INFO_{col}' for col in info_df.columns]
        
        # Combine with original DataFrame
        result_df = pd.concat([df, info_df], axis=1)
        
        logger.info(f"Extracted {len(info_df.columns)} INFO columns")
        return result_df
    
    def get_vcf_metadata(self) -> Dict[str, Any]:
        """
        Get VCF metadata from header.
        
        Returns:
            Dictionary with VCF metadata
        """
        metadata = {
            'info_fields': self.info_fields,
            'format_fields': self.format_fields,
            'header_lines': self.header_lines
        }
        
        return metadata
