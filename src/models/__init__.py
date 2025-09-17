"""Model modules for genomic variant analysis"""

from .variant_classifier import VariantClassifier
from .ensemble_model import EnsembleVariantClassifier

__all__ = [
    'VariantClassifier',
    'EnsembleVariantClassifier'
]
