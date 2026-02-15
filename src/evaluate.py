"""Model evaluation and visualization."""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from config import METRICS_PATH, PLOTS_PATH

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob))
    }
    return metrics


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model on train and test sets."""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_prob)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
    
    # Check overfitting
    auc_diff = train_metrics['roc_auc'] - test_metrics['roc_auc']
    
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'overfitting_check': {
            'train_test_auc_diff': float(auc_diff),
            'is_overfitting': bool(auc_diff > 0.05)
        }
    }
    
    # Log results
    logger.info("\nTrain Metrics:")
    for key, value in train_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nTest Metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    if auc_diff > 0.05:
        logger.warning(f"\nWarning: Train-Test AUC gap = {auc_diff:.4f}")
    
    return results, y_test_pred, y_test_prob


def save_metrics(results, model_name='logistic_regression'):
    """Save metrics to JSON file and as image."""
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    # Save JSON
    metrics_file = os.path.join(METRICS_PATH, f"{model_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nMetrics saved to {metrics_file}")
    
    # Save metrics table as image
    save_metrics_table(results, model_name)


def save_metrics_table(results, model_name='logistic_regression'):
    """Save metrics as a table image."""
    train_metrics = results['train_metrics']
    test_metrics = results['test_metrics']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    table_data = []
    
    for key, name in zip(['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'], metrics_names):
        table_data.append([
            name,
            f"{train_metrics[key]:.4f}",
            f"{test_metrics[key]:.4f}"
        ])
    
    # Add overfitting check
    auc_diff = results['overfitting_check']['train_test_auc_diff']
    overfitting_status = "Yes ⚠️" if results['overfitting_check']['is_overfitting'] else "No ✓"
    table_data.append(['', '', ''])  # Empty row
    table_data.append(['Train-Test AUC Gap', f"{auc_diff:.4f}", ''])
    table_data.append(['Overfitting Detected', overfitting_status, ''])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'Train', 'Test'],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3, 0.3]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color data rows
    for i in range(1, 6):
        for j in range(3):
            table[(i, j)].set_facecolor('#f0f0f0')
    
    # Color overfitting rows
    for i in range(7, 9):
        for j in range(3):
            if results['overfitting_check']['is_overfitting']:
                table[(i, j)].set_facecolor('#ffcccc')
            else:
                table[(i, j)].set_facecolor('#ccffcc')
    
    plt.title(f'{model_name.replace("_", " ").title()} - Performance Metrics', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save
    metrics_image = os.path.join(METRICS_PATH, f"{model_name}_metrics_table.png")
    plt.savefig(metrics_image, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Metrics table saved to {metrics_image}")


def plot_results(y_test, y_pred, y_prob, model_name='logistic_regression'):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{model_name.replace("_", " ").title()} Performance', 
                 fontsize=14, fontweight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_ylabel('True')
    axes[0,0].set_xlabel('Predicted')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0,1].plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    axes[0,1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0,1].set_title('ROC Curve')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[0,2].plot(rec, prec, lw=2)
    axes[0,2].set_title('Precision-Recall Curve')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].grid(alpha=0.3)
    
    # Prediction Distribution
    axes[1,0].hist(y_prob[y_test==0], bins=30, alpha=0.5, label='Class 0')
    axes[1,0].hist(y_prob[y_test==1], bins=30, alpha=0.5, label='Class 1')
    axes[1,0].axvline(0.5, color='k', linestyle='--')
    axes[1,0].set_title('Prediction Distribution')
    axes[1,0].set_xlabel('Predicted Probability')
    axes[1,0].legend()
    
    # Metrics Bar Chart
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    axes[1,1].bar(range(len(metrics)), list(metrics.values()))
    axes[1,1].set_xticks(range(len(metrics)))
    axes[1,1].set_xticklabels(metrics.keys(), rotation=45)
    axes[1,1].set_title('Performance Metrics')
    axes[1,1].set_ylim([0, 1.05])
    axes[1,1].grid(axis='y', alpha=0.3)
    
    # Clear last subplot
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(PLOTS_PATH, exist_ok=True)
    plot_file = os.path.join(PLOTS_PATH, f"{model_name}_results.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Plots saved to {plot_file}")
    
    plt.close()