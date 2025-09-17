import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger
from Bio.Seq import Seq
from Bio import motifs
from collections import Counter

class SequenceFeatureExtractor:
    """Extract sequence-based features from genomic data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context_size = config['data']['features']['sequence_context_size']
    
    def extract_motif_features(
        self,
        sequences: List[str],
        motif_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract motif occurrence features from sequences
        
        Args:
            sequences: List of DNA sequences
            motif_list: List of motifs to search for
            
        Returns:
            DataFrame with motif features
        """
        logger.info("Extracting motif features")
        
        if motif_list is None:
            # Common regulatory motifs
            motif_list = [
                'TATAAA',  # TATA box
                'CAAT',    # CAAT box
                'GGGCGG',  # GC box
                'CACGTG',  # E-box
                'TGACGTCA', # CRE
                'GGAA',    # ETS binding site
            ]
        
        features = []
        for seq in sequences:
            seq_upper = seq.upper()
            motif_counts = {}
            
            for motif in motif_list:
                motif_counts[f'motif_{motif}'] = seq_upper.count(motif)
                # Also count reverse complement
                rc_motif = str(Seq(motif).reverse_complement())
                motif_counts[f'motif_{motif}_rc'] = seq_upper.count(rc_motif)
            
            features.append(motif_counts)
        
        return pd.DataFrame(features)
    
    def extract_secondary_structure_features(
        self,
        sequences: List[str]
    ) -> pd.DataFrame:
        """
        Predict and extract secondary structure features
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            DataFrame with secondary structure features
        """
        logger.info("Extracting secondary structure features")
        
        features = []
        for seq in sequences:
            seq_obj = Seq(seq)
            
            # Calculate melting temperature
            from Bio.SeqUtils import MeltingTemp as mt
            
            feat = {
                'tm_wallace': mt.Tm_Wallace(seq_obj),
                'tm_gc': mt.Tm_GC(seq_obj),
                'gc_skew': self._calculate_gc_skew(seq),
                'at_skew': self._calculate_at_skew(seq),
                'purine_percent': (seq.upper().count('A') + seq.upper().count('G')) / len(seq) if seq else 0,
                'has_palindrome': self._has_palindrome(seq),
                'max_homopolymer': self._max_homopolymer_length(seq)
            }
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def extract_codon_features(
        self,
        sequences: List[str],
        reading_frame: int = 0
    ) -> pd.DataFrame:
        """
        Extract codon usage features
        
        Args:
            sequences: List of DNA sequences
            reading_frame: Reading frame offset (0, 1, or 2)
            
        Returns:
            DataFrame with codon features
        """
        logger.info("Extracting codon features")
        
        # Standard genetic code
        from Bio.Data import CodonTable
        standard_table = CodonTable.standard_dna_table
        
        features = []
        for seq in sequences:
            # Extract codons based on reading frame
            codons = [seq[i:i+3] for i in range(reading_frame, len(seq)-2, 3)]
            
            # Count codon usage
            codon_counts = Counter(codons)
            
            # Calculate codon bias metrics
            feat = {
                'num_codons': len(codons),
                'unique_codons': len(set(codons)),
                'stop_codons': sum(1 for c in codons if c in ['TAA', 'TAG', 'TGA']),
                'start_codons': codons.count('ATG') if codons else 0,
            }
            
            # Add individual codon frequencies for common codons
            common_codons = ['ATG', 'AAA', 'AAG', 'GAA', 'GAG', 'CAA', 'CAG']
            for codon in common_codons:
                feat[f'codon_{codon}_freq'] = codon_counts.get(codon, 0) / len(codons) if codons else 0
            
            # Calculate CAI (Codon Adaptation Index) - simplified version
            feat['codon_diversity'] = len(set(codons)) / len(codons) if codons else 0
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def extract_regulatory_features(
        self,
        sequences: List[str],
        upstream_length: int = 100
    ) -> pd.DataFrame:
        """
        Extract regulatory element features
        
        Args:
            sequences: List of DNA sequences
            upstream_length: Length of upstream region to analyze
            
        Returns:
            DataFrame with regulatory features
        """
        logger.info("Extracting regulatory features")
        
        features = []
        
        # Common transcription factor binding sites (simplified)
        tf_sites = {
            'SP1': 'GGGCGG',
            'AP1': 'TGACTCA',
            'NFkB': 'GGGACTTTCC',
            'CREB': 'TGACGTCA',
            'p53': 'RRRCWWGYYY',  # R=A/G, W=A/T, Y=C/T
            'STAT': 'TTCNNNGAA',  # N=any
        }
        
        for seq in sequences:
            feat = {}
            
            # Analyze upstream region
            upstream_seq = seq[:upstream_length] if len(seq) >= upstream_length else seq
            
            # Check for TFBS
            for tf_name, tf_motif in tf_sites.items():
                # Simplified motif matching (in practice, use PWMs)
                feat[f'tfbs_{tf_name}'] = self._fuzzy_motif_search(upstream_seq, tf_motif)
            
            # CpG island features
            feat['cpg_obs_exp'] = self._calculate_cpg_obs_exp(seq)
            feat['is_cpg_island'] = feat['cpg_obs_exp'] > 0.6
            
            # Repeat elements
            feat['has_tandem_repeat'] = self._has_tandem_repeat(seq)
            feat['repeat_density'] = self._calculate_repeat_density(seq)
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _calculate_gc_skew(self, seq: str) -> float:
        """Calculate GC skew: (G-C)/(G+C)"""
        g_count = seq.upper().count('G')
        c_count = seq.upper().count('C')
        gc_sum = g_count + c_count
        return (g_count - c_count) / gc_sum if gc_sum > 0 else 0
    
    def _calculate_at_skew(self, seq: str) -> float:
        """Calculate AT skew: (A-T)/(A+T)"""
        a_count = seq.upper().count('A')
        t_count = seq.upper().count('T')
        at_sum = a_count + t_count
        return (a_count - t_count) / at_sum if at_sum > 0 else 0
    
    def _has_palindrome(self, seq: str, min_length: int = 6) -> bool:
        """Check if sequence contains palindromic regions"""
        seq_upper = seq.upper()
        seq_obj = Seq(seq_upper)
        rev_comp = str(seq_obj.reverse_complement())
        
        for i in range(len(seq) - min_length + 1):
            substr = seq_upper[i:i+min_length]
            if substr in rev_comp:
                return True
        return False
    
    def _max_homopolymer_length(self, seq: str) -> int:
        """Find maximum homopolymer length"""
        if not seq:
            return 0
        
        max_length = 1
        current_length = 1
        
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1
        
        return max_length
    
    def _fuzzy_motif_search(self, seq: str, motif: str) -> int:
        """Simple fuzzy motif search (counts exact matches)"""
        # In practice, use position weight matrices
        count = 0
        motif_clean = motif.replace('N', '.')  # N means any nucleotide
        
        # For now, just count exact matches
        for base in ['A', 'C', 'G', 'T']:
            motif_instance = motif.replace('R', '[AG]').replace('W', '[AT]').replace('Y', '[CT]')
        
        return seq.upper().count(motif) if 'N' not in motif and 'R' not in motif else 0
    
    def _calculate_cpg_obs_exp(self, seq: str) -> float:
        """Calculate CpG observed/expected ratio"""
        seq_upper = seq.upper()
        
        c_count = seq_upper.count('C')
        g_count = seq_upper.count('G')
        cg_count = seq_upper.count('CG')
        
        length = len(seq)
        
        if length == 0 or c_count == 0 or g_count == 0:
            return 0
        
        expected = (c_count * g_count) / length
        
        return cg_count / expected if expected > 0 else 0
    
    def _has_tandem_repeat(self, seq: str, min_unit: int = 2, min_copies: int = 3) -> bool:
        """Check for tandem repeats"""
        seq_upper = seq.upper()
        
        for unit_length in range(min_unit, len(seq) // min_copies + 1):
            for i in range(len(seq) - unit_length * min_copies + 1):
                unit = seq_upper[i:i+unit_length]
                if unit * min_copies in seq_upper:
                    return True
        
        return False
    
    def _calculate_repeat_density(self, seq: str) -> float:
        """Calculate the density of repetitive elements"""
        if not seq:
            return 0
        
        # Simple repeat detection
        repeat_count = 0
        window = 10
        
        for i in range(len(seq) - window):
            window_seq = seq[i:i+window]
            if window_seq.count(window_seq[0]) > window * 0.7:  # >70% same nucleotide
                repeat_count += 1
        
        return repeat_count / len(seq) if seq else 0