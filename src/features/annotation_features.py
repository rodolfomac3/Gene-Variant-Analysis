"""
Annotation-based feature engineering for gene variant analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)


class AnnotationFeatureExtractor:
    """
    Feature engineering class for annotation-based features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AnnotationFeatureExtractor."""
        self.config = config
        self.feature_config = config.get('features', {})
    
    def extract_gene_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract gene-related features.
        
        Args:
            data: Input variant data with gene annotations
            
        Returns:
            Data with gene features
        """
        logger.info("Extracting gene features")
        
        feature_data = data.copy()
        
        # Gene name features
        if 'gene_name' in feature_data.columns:
            feature_data['has_gene_name'] = feature_data['gene_name'].notna().astype(int)
            feature_data['gene_name_length'] = feature_data['gene_name'].str.len().fillna(0)
        
        # Transcript features
        if 'transcript_id' in feature_data.columns:
            feature_data['has_transcript'] = feature_data['transcript_id'].notna().astype(int)
            feature_data['n_transcripts'] = feature_data.groupby('gene_name')['transcript_id'].transform('nunique').fillna(0)
        
        # Consequence features
        if 'consequence' in feature_data.columns:
            feature_data = self._extract_consequence_features(feature_data)
        
        # Impact features
        if 'impact' in feature_data.columns:
            feature_data = self._extract_impact_features(feature_data)
        
        logger.info("Gene features extracted")
        return feature_data
    
    def extract_functional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract functional annotation features.
        
        Args:
            data: Input variant data
            
        Returns:
            Data with functional features
        """
        logger.info("Extracting functional features")
        
        feature_data = data.copy()
        
        # Protein domain features
        if 'protein_domain' in feature_data.columns:
            feature_data['has_protein_domain'] = feature_data['protein_domain'].notna().astype(int)
            feature_data['n_protein_domains'] = feature_data['protein_domain'].str.count(';').fillna(0) + 1
        
        # SIFT and PolyPhen scores
        if 'sift_score' in feature_data.columns:
            feature_data['sift_deleterious'] = (feature_data['sift_score'] < 0.05).astype(int)
            feature_data['sift_tolerated'] = (feature_data['sift_score'] > 0.05).astype(int)
        
        if 'polyphen_score' in feature_data.columns:
            feature_data['polyphen_deleterious'] = (feature_data['polyphen_score'] > 0.5).astype(int)
            feature_data['polyphen_benign'] = (feature_data['polyphen_score'] < 0.5).astype(int)
        
        # CADD score features
        if 'cadd_score' in feature_data.columns:
            feature_data['cadd_high'] = (feature_data['cadd_score'] > 15).astype(int)
            feature_data['cadd_very_high'] = (feature_data['cadd_score'] > 20).astype(int)
        
        logger.info("Functional features extracted")
        return feature_data
    
    def extract_pathway_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pathway-related features.
        
        Args:
            data: Input variant data
            
        Returns:
            Data with pathway features
        """
        logger.info("Extracting pathway features")
        
        feature_data = data.copy()
        
        # GO terms
        if 'go_terms' in feature_data.columns:
            feature_data['has_go_terms'] = feature_data['go_terms'].notna().astype(int)
            feature_data['n_go_terms'] = feature_data['go_terms'].str.count(';').fillna(0) + 1
        
        # KEGG pathways
        if 'kegg_pathways' in feature_data.columns:
            feature_data['has_kegg_pathways'] = feature_data['kegg_pathways'].notna().astype(int)
            feature_data['n_kegg_pathways'] = feature_data['kegg_pathways'].str.count(';').fillna(0) + 1
        
        # Reactome pathways
        if 'reactome_pathways' in feature_data.columns:
            feature_data['has_reactome_pathways'] = feature_data['reactome_pathways'].notna().astype(int)
            feature_data['n_reactome_pathways'] = feature_data['reactome_pathways'].str.count(';').fillna(0) + 1
        
        logger.info("Pathway features extracted")
        return feature_data
    
    def extract_expression_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract gene expression features.
        
        Args:
            data: Input variant data
            
        Returns:
            Data with expression features
        """
        logger.info("Extracting expression features")
        
        feature_data = data.copy()
        
        # Tissue expression levels (simulated)
        tissues = ['brain', 'liver', 'heart', 'lung', 'kidney', 'muscle']
        for tissue in tissues:
            col_name = f'expression_{tissue}'
            if col_name not in feature_data.columns:
                feature_data[col_name] = np.random.uniform(0, 10, len(feature_data))
            
            feature_data[f'{tissue}_highly_expressed'] = (feature_data[col_name] > 5).astype(int)
            feature_data[f'{tissue}_lowly_expressed'] = (feature_data[col_name] < 2).astype(int)
        
        # Overall expression metrics
        expression_cols = [f'expression_{tissue}' for tissue in tissues]
        if all(col in feature_data.columns for col in expression_cols):
            feature_data['max_expression'] = feature_data[expression_cols].max(axis=1)
            feature_data['mean_expression'] = feature_data[expression_cols].mean(axis=1)
            feature_data['expression_variance'] = feature_data[expression_cols].var(axis=1)
        
        logger.info("Expression features extracted")
        return feature_data
    
    def _extract_consequence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract consequence-based features."""
        # Common consequence types
        consequence_types = [
            'synonymous_variant', 'missense_variant', 'nonsense_variant',
            'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant',
            'stop_gained', 'stop_lost', 'start_lost', 'inframe_insertion',
            'inframe_deletion', 'protein_altering_variant'
        ]
        
        for cons_type in consequence_types:
            data[f'is_{cons_type}'] = data['consequence'].str.contains(cons_type, na=False).astype(int)
        
        # Consequence severity
        def get_consequence_severity(consequence):
            if pd.isna(consequence):
                return 'unknown'
            
            consequence = str(consequence).lower()
            
            if any(term in consequence for term in ['nonsense', 'frameshift', 'stop_gained', 'stop_lost']):
                return 'high'
            elif any(term in consequence for term in ['missense', 'splice', 'protein_altering']):
                return 'moderate'
            elif any(term in consequence for term in ['synonymous', 'intron']):
                return 'low'
            else:
                return 'modifier'
        
        data['consequence_severity'] = data['consequence'].apply(get_consequence_severity)
        
        # Binary severity features
        severity_levels = ['high', 'moderate', 'low', 'modifier', 'unknown']
        for severity in severity_levels:
            data[f'severity_{severity}'] = (data['consequence_severity'] == severity).astype(int)
        
        return data
    
    def _extract_impact_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract impact-based features."""
        # Impact levels
        impact_levels = ['HIGH', 'MODERATE', 'LOW', 'MODIFIER']
        
        for impact in impact_levels:
            data[f'impact_{impact.lower()}'] = (data['impact'] == impact).astype(int)
        
        # Impact severity score
        impact_scores = {'HIGH': 4, 'MODERATE': 3, 'LOW': 2, 'MODIFIER': 1}
        data['impact_score'] = data['impact'].map(impact_scores).fillna(0)
        
        return data
