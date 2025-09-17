"""
Annotation utilities for gene variant analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)


class AnnotationUtils:
    """
    Utilities for handling genomic annotations.
    """
    
    def __init__(self):
        """Initialize AnnotationUtils."""
        pass
    
    def parse_vep_annotation(self, vep_string: str) -> Dict[str, Any]:
        """
        Parse VEP (Variant Effect Predictor) annotation string.
        
        Args:
            vep_string: VEP annotation string
            
        Returns:
            Dictionary with parsed VEP data
        """
        if pd.isna(vep_string) or vep_string == '.':
            return {}
        
        # VEP format: Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|Existing_variation|DISTANCE|STRAND|FLAGS|VARIANT_CLASS|SYMBOL_SOURCE|HGNC_ID|CANONICAL|MANE|TSL|APPRIS|CCDS|ENSP|SWISSPROT|TREMBL|UNIPARC|GENE_PENDING|HGVS_OFFSET|MOTIF_NAME|MOTIF_POS|HIGH_INF_POS|MOTIF_SCORE_CHANGE
        
        fields = vep_string.split('|')
        
        vep_data = {
            'allele': fields[0] if len(fields) > 0 else '',
            'consequence': fields[1] if len(fields) > 1 else '',
            'impact': fields[2] if len(fields) > 2 else '',
            'symbol': fields[3] if len(fields) > 3 else '',
            'gene': fields[4] if len(fields) > 4 else '',
            'feature_type': fields[5] if len(fields) > 5 else '',
            'feature': fields[6] if len(fields) > 6 else '',
            'biotype': fields[7] if len(fields) > 7 else '',
            'exon': fields[8] if len(fields) > 8 else '',
            'intron': fields[9] if len(fields) > 9 else '',
            'hgvsc': fields[10] if len(fields) > 10 else '',
            'hgvsp': fields[11] if len(fields) > 11 else '',
            'cdna_position': fields[12] if len(fields) > 12 else '',
            'cds_position': fields[13] if len(fields) > 13 else '',
            'protein_position': fields[14] if len(fields) > 14 else '',
            'amino_acids': fields[15] if len(fields) > 15 else '',
            'codons': fields[16] if len(fields) > 16 else '',
            'existing_variation': fields[17] if len(fields) > 17 else '',
            'distance': fields[18] if len(fields) > 18 else '',
            'strand': fields[19] if len(fields) > 19 else '',
            'flags': fields[20] if len(fields) > 20 else '',
            'variant_class': fields[21] if len(fields) > 21 else '',
            'symbol_source': fields[22] if len(fields) > 22 else '',
            'hgnc_id': fields[23] if len(fields) > 23 else '',
            'canonical': fields[24] if len(fields) > 24 else '',
            'mane': fields[25] if len(fields) > 25 else '',
            'tsl': fields[26] if len(fields) > 26 else '',
            'appris': fields[27] if len(fields) > 27 else '',
            'ccds': fields[28] if len(fields) > 28 else '',
            'ensp': fields[29] if len(fields) > 29 else '',
            'swissprot': fields[30] if len(fields) > 30 else '',
            'trembl': fields[31] if len(fields) > 31 else '',
            'uniparc': fields[32] if len(fields) > 32 else '',
            'gene_pending': fields[33] if len(fields) > 33 else '',
            'hgvs_offset': fields[34] if len(fields) > 34 else '',
            'motif_name': fields[35] if len(fields) > 35 else '',
            'motif_pos': fields[36] if len(fields) > 36 else '',
            'high_inf_pos': fields[37] if len(fields) > 37 else '',
            'motif_score_change': fields[38] if len(fields) > 38 else ''
        }
        
        return vep_data
    
    def parse_annovar_annotation(self, annovar_string: str) -> Dict[str, Any]:
        """
        Parse ANNOVAR annotation string.
        
        Args:
            annovar_string: ANNOVAR annotation string
            
        Returns:
            Dictionary with parsed ANNOVAR data
        """
        if pd.isna(annovar_string) or annovar_string == '.':
            return {}
        
        # ANNOVAR format varies by database, this is a generic parser
        # Format: gene_name|transcript_id|consequence|impact|...
        
        fields = annovar_string.split('|')
        
        annovar_data = {
            'gene_name': fields[0] if len(fields) > 0 else '',
            'transcript_id': fields[1] if len(fields) > 1 else '',
            'consequence': fields[2] if len(fields) > 2 else '',
            'impact': fields[3] if len(fields) > 3 else '',
            'additional_info': fields[4:] if len(fields) > 4 else []
        }
        
        return annovar_data
    
    def extract_consequence_severity(self, consequence: str) -> str:
        """
        Extract consequence severity from consequence string.
        
        Args:
            consequence: Consequence string
            
        Returns:
            Severity level (high, moderate, low, modifier)
        """
        if pd.isna(consequence):
            return 'unknown'
        
        consequence = str(consequence).lower()
        
        # High impact consequences
        high_impact = ['nonsense', 'frameshift', 'stop_gained', 'stop_lost', 
                      'start_lost', 'splice_acceptor', 'splice_donor']
        
        # Moderate impact consequences
        moderate_impact = ['missense', 'inframe_insertion', 'inframe_deletion',
                          'protein_altering', 'splice_region']
        
        # Low impact consequences
        low_impact = ['synonymous', 'intron', 'upstream', 'downstream',
                     'intergenic', 'utr']
        
        if any(term in consequence for term in high_impact):
            return 'high'
        elif any(term in consequence for term in moderate_impact):
            return 'moderate'
        elif any(term in consequence for term in low_impact):
            return 'low'
        else:
            return 'modifier'
    
    def extract_impact_score(self, impact: str) -> int:
        """
        Extract numeric impact score from impact string.
        
        Args:
            impact: Impact string (HIGH, MODERATE, LOW, MODIFIER)
            
        Returns:
            Numeric impact score
        """
        impact_scores = {
            'HIGH': 4,
            'MODERATE': 3,
            'LOW': 2,
            'MODIFIER': 1
        }
        
        return impact_scores.get(str(impact).upper(), 0)
    
    def parse_cadd_scores(self, cadd_string: str) -> Dict[str, float]:
        """
        Parse CADD scores from annotation string.
        
        Args:
            cadd_string: CADD annotation string
            
        Returns:
            Dictionary with CADD scores
        """
        if pd.isna(cadd_string) or cadd_string == '.':
            return {}
        
        cadd_data = {}
        
        # CADD format: CADD_RAW=score,CADD_PHRED=score
        if '=' in cadd_string:
            parts = cadd_string.split(',')
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    try:
                        cadd_data[key.strip()] = float(value.strip())
                    except ValueError:
                        cadd_data[key.strip()] = value.strip()
        
        return cadd_data
    
    def parse_sift_polyphen_scores(self, scores_string: str) -> Dict[str, Any]:
        """
        Parse SIFT and PolyPhen scores from annotation string.
        
        Args:
            scores_string: SIFT/PolyPhen annotation string
            
        Returns:
            Dictionary with SIFT and PolyPhen scores
        """
        if pd.isna(scores_string) or scores_string == '.':
            return {}
        
        scores_data = {}
        
        # Format: SIFT=score(prediction),PolyPhen=score(prediction)
        if '=' in scores_string:
            parts = scores_string.split(',')
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    
                    # Parse score and prediction
                    if '(' in value and ')' in value:
                        score_str, prediction = value.split('(', 1)
                        prediction = prediction.rstrip(')')
                        try:
                            scores_data[f'{key}_score'] = float(score_str.strip())
                        except ValueError:
                            scores_data[f'{key}_score'] = score_str.strip()
                        scores_data[f'{key}_prediction'] = prediction.strip()
                    else:
                        try:
                            scores_data[key] = float(value.strip())
                        except ValueError:
                            scores_data[key] = value.strip()
        
        return scores_data
    
    def create_annotation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary of annotations in the dataset.
        
        Args:
            df: DataFrame with annotation data
            
        Returns:
            Dictionary with annotation summary
        """
        logger.info("Creating annotation summary")
        
        summary = {
            'total_variants': len(df),
            'consequence_summary': {},
            'impact_summary': {},
            'gene_summary': {},
            'feature_summary': {}
        }
        
        # Consequence summary
        if 'consequence' in df.columns:
            consequence_counts = df['consequence'].value_counts()
            summary['consequence_summary'] = {
                'unique_consequences': len(consequence_counts),
                'top_consequences': consequence_counts.head(10).to_dict()
            }
        
        # Impact summary
        if 'impact' in df.columns:
            impact_counts = df['impact'].value_counts()
            summary['impact_summary'] = {
                'unique_impacts': len(impact_counts),
                'impact_distribution': impact_counts.to_dict()
            }
        
        # Gene summary
        if 'gene_name' in df.columns:
            gene_counts = df['gene_name'].value_counts()
            summary['gene_summary'] = {
                'unique_genes': len(gene_counts),
                'top_genes': gene_counts.head(10).to_dict()
            }
        
        # Feature summary
        if 'feature_type' in df.columns:
            feature_counts = df['feature_type'].value_counts()
            summary['feature_summary'] = {
                'unique_features': len(feature_counts),
                'feature_distribution': feature_counts.to_dict()
            }
        
        logger.info("Annotation summary created")
        return summary
