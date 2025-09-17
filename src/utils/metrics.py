"""
Metrics and evaluation utilities for gene variant analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from config.logging_config import get_logger

logger = get_logger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating classification metrics")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
    
    logger.info(f"Metrics calculated: {metrics}")
    return metrics


def plot_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                y_pred_proba: Optional[np.ndarray] = None,
                save_path: Optional[str] = None) -> None:
    """
    Plot evaluation metrics and visualizations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        save_path: Path to save plots (optional)
    """
    logger.info("Creating evaluation plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt='.3f', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Classification Report')
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc="lower right")
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        axes[1, 1].plot(recall, precision, color='darkorange', lw=2, 
                       label=f'PR curve (AP = {avg_precision:.2f})')
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].legend(loc="lower left")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {save_path}")
    
    plt.show()


def calculate_feature_importance_metrics(feature_importance: pd.DataFrame, 
                                       top_n: int = 20) -> Dict[str, Any]:
    """
    Calculate feature importance metrics.
    
    Args:
        feature_importance: DataFrame with feature importance
        top_n: Number of top features to analyze
        
    Returns:
        Dictionary with feature importance metrics
    """
    logger.info("Calculating feature importance metrics")
    
    # Sort by importance
    sorted_features = feature_importance.sort_values('importance', ascending=False)
    
    metrics = {
        'top_features': sorted_features.head(top_n).to_dict('records'),
        'total_features': len(feature_importance),
        'importance_stats': {
            'mean': feature_importance['importance'].mean(),
            'std': feature_importance['importance'].std(),
            'min': feature_importance['importance'].min(),
            'max': feature_importance['importance'].max()
        },
        'top_n_features': top_n
    }
    
    # Calculate cumulative importance
    sorted_features['cumulative_importance'] = sorted_features['importance'].cumsum()
    metrics['cumulative_importance'] = sorted_features['cumulative_importance'].to_dict()
    
    logger.info("Feature importance metrics calculated")
    return metrics


def plot_feature_importance(feature_importance: pd.DataFrame, 
                          top_n: int = 20,
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_importance: DataFrame with feature importance
        top_n: Number of top features to plot
        save_path: Path to save plot (optional)
    """
    logger.info("Creating feature importance plot")
    
    # Get top N features
    top_features = feature_importance.nlargest(top_n, 'importance')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def calculate_model_performance_summary(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Calculate model performance summary with qualitative assessment.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary with performance summary
    """
    logger.info("Calculating model performance summary")
    
    summary = {}
    
    # Accuracy assessment
    accuracy = metrics.get('accuracy', 0)
    if accuracy >= 0.9:
        summary['accuracy'] = 'Excellent'
    elif accuracy >= 0.8:
        summary['accuracy'] = 'Good'
    elif accuracy >= 0.7:
        summary['accuracy'] = 'Fair'
    else:
        summary['accuracy'] = 'Poor'
    
    # F1 Score assessment
    f1 = metrics.get('f1_score', 0)
    if f1 >= 0.9:
        summary['f1_score'] = 'Excellent'
    elif f1 >= 0.8:
        summary['f1_score'] = 'Good'
    elif f1 >= 0.7:
        summary['f1_score'] = 'Fair'
    else:
        summary['f1_score'] = 'Poor'
    
    # ROC AUC assessment (if available)
    if 'roc_auc' in metrics:
        roc_auc = metrics['roc_auc']
        if roc_auc >= 0.9:
            summary['roc_auc'] = 'Excellent'
        elif roc_auc >= 0.8:
            summary['roc_auc'] = 'Good'
        elif roc_auc >= 0.7:
            summary['roc_auc'] = 'Fair'
        else:
            summary['roc_auc'] = 'Poor'
    
    logger.info(f"Performance summary: {summary}")
    return summary
