"""
Variant-specific feature engineering for gene variant analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)


class VariantFeatureExtractor:
    """
    Feature engineering class for variant-specific features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize VariantFeatureExtractor."""
        self.config = config
        self.feature_config = config.get('features', {})
    
    def extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic variant features.
        
        Args:
            data: Input variant data
            
        Returns:
            Data with basic variant features
        """
        logger.info("Extracting basic variant features")
        
        feature_data = data.copy()
        
        # Extract variant type
        feature_data = self._extract_variant_type(feature_data)
        
        # Extract allele length features
        feature_data = self._extract_allele_length_features(feature_data)
        
        # Extract nucleotide composition features
        feature_data = self._extract_nucleotide_composition_features(feature_data)
        
        # Extract transition/transversion features
        feature_data = self._extract_transition_transversion_features(feature_data)
        
        # Extract position features
        feature_data = self._extract_position_features(feature_data)
        
        logger.info(f"Basic features extracted. Final shape: {feature_data.shape}")
        return feature_data
    
    def extract_conservation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract conservation-based features.
        
        Args:
            data: Input variant data
            
        Returns:
            Data with conservation features
        """
        logger.info("Extracting conservation features")
        
        feature_data = data.copy()
        
        # Simulate conservation scores (in practice, load from PhyloP, PhastCons, etc.)
        if 'conservation_score' not in feature_data.columns:
            feature_data['conservation_score'] = np.random.uniform(0, 1, len(feature_data))
        
        # Conservation categories
        feature_data['is_conserved'] = (feature_data['conservation_score'] > 0.5).astype(int)
        feature_data['is_highly_conserved'] = (feature_data['conservation_score'] > 0.8).astype(int)
        
        logger.info("Conservation features extracted")
        return feature_data
    
    def extract_population_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract population frequency features.
        
        Args:
            data: Input variant data
            
        Returns:
            Data with population features
        """
        logger.info("Extracting population features")
        
        feature_data = data.copy()
        
        # Simulate population frequencies (in practice, load from gnomAD, 1000 Genomes, etc.)
        if 'af_gnomad' not in feature_data.columns:
            feature_data['af_gnomad'] = np.random.uniform(0, 0.5, len(feature_data))
        
        if 'af_1kg' not in feature_data.columns:
            feature_data['af_1kg'] = np.random.uniform(0, 0.5, len(feature_data))
        
        # Population frequency categories
        feature_data['is_rare'] = (feature_data['af_gnomad'] < 0.01).astype(int)
        feature_data['is_common'] = (feature_data['af_gnomad'] > 0.05).astype(int)
        
        # Combined frequency
        feature_data['max_af'] = feature_data[['af_gnomad', 'af_1kg']].max(axis=1)
        
        logger.info("Population features extracted")
        return feature_data
    
    def _extract_variant_type(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract variant type features."""
        if 'REF' not in data.columns or 'ALT' not in data.columns:
            logger.warning("REF or ALT columns not found, skipping variant type extraction")
            return data
        
        def classify_variant_type(row):
            ref_len = len(str(row['REF']))
            alt_len = len(str(row['ALT']))
            
            if ref_len == 1 and alt_len == 1:
                return 'SNV'
            elif ref_len > alt_len:
                return 'Deletion'
            elif ref_len < alt_len:
                return 'Insertion'
            else:
                return 'Complex'
        
        data['variant_type'] = data.apply(classify_variant_type, axis=1)
        
        # Create binary features for each variant type
        variant_types = ['SNV', 'Deletion', 'Insertion', 'Complex']
        for vt in variant_types:
            data[f'is_{vt.lower()}'] = (data['variant_type'] == vt).astype(int)
        
        return data
    
    def _extract_allele_length_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract allele length-based features."""
        if 'REF' not in data.columns or 'ALT' not in data.columns:
            return data
        
        data['ref_length'] = data['REF'].str.len()
        data['alt_length'] = data['ALT'].str.len()
        data['length_difference'] = data['alt_length'] - data['ref_length']
        data['max_allele_length'] = data[['ref_length', 'alt_length']].max(axis=1)
        data['min_allele_length'] = data[['ref_length', 'alt_length']].min(axis=1)
        
        return data
    
    def _extract_nucleotide_composition_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract nucleotide composition features."""
        if 'REF' not in data.columns or 'ALT' not in data.columns:
            return data
        
        def get_nucleotide_counts(sequence):
            sequence = str(sequence).upper()
            return {
                'A': sequence.count('A'),
                'T': sequence.count('T'),
                'G': sequence.count('G'),
                'C': sequence.count('C')
            }
        
        # Reference allele composition
        ref_composition = data['REF'].apply(get_nucleotide_counts)
        ref_comp_df = pd.DataFrame(ref_composition.tolist(), index=data.index)
        ref_comp_df.columns = [f'ref_{col}' for col in ref_comp_df.columns]
        data = pd.concat([data, ref_comp_df], axis=1)
        
        # Alternate allele composition
        alt_composition = data['ALT'].apply(get_nucleotide_counts)
        alt_comp_df = pd.DataFrame(alt_composition.tolist(), index=data.index)
        alt_comp_df.columns = [f'alt_{col}' for col in alt_comp_df.columns]
        data = pd.concat([data, alt_comp_df], axis=1)
        
        # GC content features
        data['ref_gc_content'] = (data['ref_G'] + data['ref_C']) / (
            data['ref_A'] + data['ref_T'] + data['ref_G'] + data['ref_C'] + 1e-8
        )
        data['alt_gc_content'] = (data['alt_G'] + data['alt_C']) / (
            data['alt_A'] + data['alt_T'] + data['alt_G'] + data['alt_C'] + 1e-8
        )
        data['gc_content_difference'] = data['alt_gc_content'] - data['ref_gc_content']
        
        return data
    
    def _extract_transition_transversion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract transition/transversion features for SNVs."""
        if 'REF' not in data.columns or 'ALT' not in data.columns:
            return data
        
        def classify_mutation_type(row):
            ref = str(row['REF']).upper()
            alt = str(row['ALT']).upper()
            
            if len(ref) != 1 or len(alt) != 1:
                return 'Not_SNV'
            
            # Transitions: A<->G, C<->T
            transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
            
            if (ref, alt) in transitions:
                return 'Transition'
            else:
                return 'Transversion'
        
        data['mutation_type'] = data.apply(classify_mutation_type, axis=1)
        data['is_transition'] = (data['mutation_type'] == 'Transition').astype(int)
        data['is_transversion'] = (data['mutation_type'] == 'Transversion').astype(int)
        
        return data
    
    def _extract_position_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract position-based features."""
        if 'POS' not in data.columns:
            return data
        
        data['position'] = pd.to_numeric(data['POS'], errors='coerce')
        data['position_log'] = np.log(data['position'] + 1)
        data['position_sqrt'] = np.sqrt(data['position'])
        
        # Position bins
        data['position_bin'] = pd.cut(data['position'], bins=10, labels=False)
        
        return data
