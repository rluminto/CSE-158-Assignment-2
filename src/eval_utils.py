"""
Evaluation utilities for the Food.com recipe recommender project.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, roc_curve,
    precision_recall_curve, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List


def evaluate_classification(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Comprehensive classification evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        model_name: Name for display
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['log_loss'] = log_loss(y_true, y_proba)
        metrics['brier_score'] = brier_score_loss(y_true, y_proba)
    
    return metrics


def compare_models(results_list, metric='roc_auc'):
    """
    Compare multiple models side-by-side.
    
    Args:
        results_list: List of metric dictionaries from evaluate_classification
        metric: Metric to sort by
        
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results_list)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)
    return df


def plot_roc_curve(y_true, y_proba, model_name="Model", ax=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name for display
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    return ax


def plot_precision_recall_curve(y_true, y_proba, model_name="Model", ax=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name for display
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    ax.plot(recall, precision, label=model_name, linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    return ax


def plot_confusion_matrix(y_true, y_pred, labels=['Not Like', 'Like'], ax=None):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels for display
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    return ax


def plot_feature_importance(coefficients_df, top_n=20, ax=None):
    """
    Plot feature importance from logistic regression coefficients.
    
    Args:
        coefficients_df: DataFrame with 'feature' and 'coefficient' columns
        top_n: Number of top features to show
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get top N by absolute value
    top_features = coefficients_df.head(top_n)
    
    colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
    ax.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Coefficients (Logistic Regression)', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3, axis='x')
    
    return ax


def precision_at_k(y_true, y_pred_ranked, k=10):
    """
    Compute Precision@K for a single user.
    
    Args:
        y_true: True labels for ranked items
        y_pred_ranked: Predicted rankings (assumed sorted by score)
        k: Number of top items to consider
        
    Returns:
        Precision@K value
    """
    if len(y_pred_ranked) == 0:
        return 0.0
    
    top_k = y_true[:k]
    return top_k.sum() / min(k, len(top_k))


def recall_at_k(y_true, y_pred_ranked, k=10):
    """
    Compute Recall@K for a single user.
    
    Args:
        y_true: True labels for ranked items
        y_pred_ranked: Predicted rankings (assumed sorted by score)
        k: Number of top items to consider
        
    Returns:
        Recall@K value
    """
    if y_true.sum() == 0:
        return 0.0
    
    top_k = y_true[:k]
    return top_k.sum() / y_true.sum()


def evaluate_recommender(test_df, recommendations_dict, k_values=[5, 10], 
                        recipe_col='recipe_id', target_col='is_like'):
    """
    Evaluate recommender system using Precision@K and Recall@K.
    
    Args:
        test_df: Test DataFrame with actual interactions
        recommendations_dict: Dict mapping user_id -> DataFrame of recommendations
        k_values: List of K values to evaluate
        recipe_col: Column name for recipe ID
        target_col: Column name for target (is_like)
        
    Returns:
        Dictionary of average metrics
    """
    results = {f'precision@{k}': [] for k in k_values}
    results.update({f'recall@{k}': [] for k in k_values})
    
    for user_id, recs in recommendations_dict.items():
        # Get user's test interactions
        user_test = test_df[test_df['user_id'] == user_id]
        
        if len(user_test) == 0:
            continue
        
        # Get which recommended recipes were actually liked
        liked_recipes = set(user_test[user_test[target_col] == 1][recipe_col].values)
        rec_recipes = recs[recipe_col].values
        
        # Create binary vector: 1 if recommended recipe was liked, 0 otherwise
        y_true_ranked = np.array([1 if r in liked_recipes else 0 for r in rec_recipes])
        
        # Compute metrics for each K
        for k in k_values:
            results[f'precision@{k}'].append(precision_at_k(y_true_ranked, rec_recipes, k))
            results[f'recall@{k}'].append(recall_at_k(y_true_ranked, rec_recipes, k))
    
    # Average across users
    avg_results = {metric: np.mean(values) if values else 0.0 
                   for metric, values in results.items()}
    
    return avg_results


def analyze_health_bias(recommendations_dict, recipes_df, 
                       health_col='is_healthy', recipe_col='recipe_id'):
    """
    Analyze whether recommender is biased toward unhealthy recipes.
    
    Args:
        recommendations_dict: Dict mapping user_id -> DataFrame of recommendations
        recipes_df: DataFrame with recipe features including health indicator
        health_col: Column name for health indicator
        recipe_col: Column name for recipe ID
        
    Returns:
        Dictionary with bias metrics
    """
    all_recommended = []
    for user_id, recs in recommendations_dict.items():
        all_recommended.extend(recs[recipe_col].values)
    
    # Get health status of recommended recipes
    rec_health = recipes_df[recipes_df['id'].isin(all_recommended)][health_col]
    
    # Get health status of all recipes (as baseline)
    all_health = recipes_df[health_col]
    
    results = {
        'healthy_fraction_all_recipes': all_health.mean(),
        'healthy_fraction_recommended': rec_health.mean(),
        'bias_ratio': rec_health.mean() / (all_health.mean() + 1e-8),
        'n_recommendations': len(all_recommended),
        'n_unique_recipes_recommended': len(set(all_recommended))
    }
    
    return results


def plot_health_tradeoff(tradeoff_results, x_col='health_weight', 
                        y1_col='precision@10', y2_col='healthy_fraction'):
    """
    Plot trade-off between recommendation quality and healthiness.
    
    Args:
        tradeoff_results: DataFrame with tradeoff experiment results
        x_col: Column for x-axis (e.g., health_weight)
        y1_col: Column for left y-axis (e.g., precision)
        y2_col: Column for right y-axis (e.g., healthy fraction)
        
    Returns:
        Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot precision on left y-axis
    ax1.plot(tradeoff_results[x_col], tradeoff_results[y1_col], 
            'b-o', linewidth=2, markersize=8, label=y1_col)
    ax1.set_xlabel('Health Weight (α)', fontsize=12)
    ax1.set_ylabel(y1_col, fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(alpha=0.3)
    
    # Plot healthy fraction on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(tradeoff_results[x_col], tradeoff_results[y2_col], 
            'g-s', linewidth=2, markersize=8, label=y2_col)
    ax2.set_ylabel(y2_col, fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title('Trade-off: Recommendation Quality vs. Healthiness', fontsize=14)
    fig.tight_layout()
    
    return fig


def create_evaluation_report(train_results, val_results, test_results, 
                            save_path=None):
    """
    Create a comprehensive evaluation report.
    
    Args:
        train_results: Metrics on training set
        val_results: Metrics on validation set
        test_results: Metrics on test set
        save_path: Optional path to save report
        
    Returns:
        DataFrame with all results
    """
    # Combine results
    train_results['split'] = 'train'
    val_results['split'] = 'validation'
    test_results['split'] = 'test'
    
    report = pd.DataFrame([train_results, val_results, test_results])
    
    # Reorder columns
    col_order = ['split', 'model'] + [c for c in report.columns if c not in ['split', 'model']]
    report = report[[c for c in col_order if c in report.columns]]
    
    if save_path:
        report.to_csv(save_path, index=False)
        print(f"Evaluation report saved to {save_path}")
    
    return report
