"""
Visualization utilities for genomic variant analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)


class GenomicVisualizer:
    """
    Visualization class for genomic variant analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize GenomicVisualizer."""
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_variant_distribution(self, data: pd.DataFrame, 
                                 save_path: Optional[str] = None) -> None:
        """
        Plot distribution of variant types.
        
        Args:
            data: Variant data
            save_path: Path to save plot
        """
        logger.info("Creating variant distribution plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Variant type distribution
        if 'variant_type' in data.columns:
            variant_counts = data['variant_type'].value_counts()
            axes[0, 0].pie(variant_counts.values, labels=variant_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Variant Type Distribution')
        
        # Chromosome distribution
        if 'CHROM' in data.columns:
            chrom_counts = data['CHROM'].value_counts().head(10)
            axes[0, 1].bar(chrom_counts.index, chrom_counts.values)
            axes[0, 1].set_title('Top 10 Chromosomes')
            axes[0, 1].set_xlabel('Chromosome')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Position distribution
        if 'POS' in data.columns:
            axes[1, 0].hist(data['POS'], bins=50, alpha=0.7)
            axes[1, 0].set_title('Position Distribution')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Frequency')
        
        # Quality score distribution
        if 'QUAL' in data.columns:
            axes[1, 1].hist(data['QUAL'], bins=50, alpha=0.7)
            axes[1, 1].set_title('Quality Score Distribution')
            axes[1, 1].set_xlabel('Quality Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Variant distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance: DataFrame with feature importance
            top_n: Number of top features to plot
            save_path: Path to save plot
        """
        logger.info("Creating feature importance plot")
        
        # Get top N features
        top_features = feature_importance.nlargest(top_n, 'importance')
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame,
                               method: str = 'pearson',
                               save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix of numeric features.
        
        Args:
            data: Data with numeric features
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_path: Path to save plot
        """
        logger.info("Creating correlation matrix plot")
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning("No numeric columns found for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr(method=method)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Matrix ({method.title()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_performance(self, metrics: Dict[str, float],
                              save_path: Optional[str] = None) -> None:
        """
        Plot model performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            save_path: Path to save plot
        """
        logger.info("Creating model performance plot")
        
        # Extract metric names and values
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Model Performance Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        from sklearn.metrics import roc_curve, auc
        
        logger.info("Creating ROC curve plot")
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        logger.info("Creating precision-recall curve plot")
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, data: pd.DataFrame,
                                   save_path: Optional[str] = None) -> None:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            data: Variant data
            save_path: Path to save HTML dashboard
        """
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Variant Types', 'Chromosome Distribution', 
                          'Position Distribution', 'Quality Scores'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Variant type pie chart
        if 'variant_type' in data.columns:
            variant_counts = data['variant_type'].value_counts()
            fig.add_trace(
                go.Pie(labels=variant_counts.index, values=variant_counts.values),
                row=1, col=1
            )
        
        # Chromosome bar chart
        if 'CHROM' in data.columns:
            chrom_counts = data['CHROM'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=chrom_counts.index, y=chrom_counts.values),
                row=1, col=2
            )
        
        # Position histogram
        if 'POS' in data.columns:
            fig.add_trace(
                go.Histogram(x=data['POS'], nbinsx=50),
                row=2, col=1
            )
        
        # Quality score histogram
        if 'QUAL' in data.columns:
            fig.add_trace(
                go.Histogram(x=data['QUAL'], nbinsx=50),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Genomic Variant Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def plot_data_quality_report(self, quality_report: Dict[str, Any],
                                save_path: Optional[str] = None) -> None:
        """
        Plot data quality report.
        
        Args:
            quality_report: Data quality report dictionary
            save_path: Path to save plot
        """
        logger.info("Creating data quality report plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Missing data percentage
        missing_pct = quality_report.get('missing_data', {}).get('missing_percentage', 0)
        axes[0, 0].bar(['Missing Data'], [missing_pct], color='red', alpha=0.7)
        axes[0, 0].set_title('Missing Data Percentage')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].set_ylim(0, 100)
        
        # Duplicate data percentage
        duplicate_pct = quality_report.get('duplicate_data', {}).get('duplicate_percentage', 0)
        axes[0, 1].bar(['Duplicate Data'], [duplicate_pct], color='orange', alpha=0.7)
        axes[0, 1].set_title('Duplicate Data Percentage')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].set_ylim(0, 100)
        
        # Data types distribution
        data_types = quality_report.get('data_types', {})
        if data_types:
            type_counts = pd.Series(data_types).value_counts()
            axes[1, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Data Types Distribution')
        
        # Basic statistics
        basic_stats = quality_report.get('basic_stats', {})
        if basic_stats:
            stats_names = list(basic_stats.keys())
            stats_values = list(basic_stats.values())
            axes[1, 1].bar(stats_names, stats_values, color='green', alpha=0.7)
            axes[1, 1].set_title('Basic Statistics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data quality report plot saved to {save_path}")
        
        plt.show()
