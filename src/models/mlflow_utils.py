"""
MLflow utilities for ChurnGuard AI.

Provides helper functions for experiment tracking, logging, and model management.
"""

import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)


class MLflowExperimentTracker:
    """
    Manages MLflow experiment tracking for ChurnGuard AI.
    
    Handles experiment setup, run logging, and artifact management.
    
    Example:
        tracker = MLflowExperimentTracker()
        tracker.start_run("logistic-baseline-v1")
        tracker.log_params({"C": 1.0, "penalty": "l2"})
        tracker.log_metrics({"accuracy": 0.87, "roc_auc": 0.92})
        tracker.log_model(model, "model")
        tracker.end_run()
    """
    
    def __init__(self, config_path: str = "configs/mlflow_config.yaml"):
        """
        Initialize MLflow tracker.
        
        Args:
            config_path: Path to MLflow configuration YAML
        """
        self.config = self._load_config(config_path)
        self._setup_mlflow()
        self.current_run = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load MLflow configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.config['tracking']['uri'])
        
        # Set or create experiment
        experiment_name = self.config['tracking']['experiment_name']
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags=self.config['tracking']['experiment_tags']
                )
                print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
            else:
                mlflow.set_experiment(experiment_name)
                print(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            print(f"Warning: Could not set up experiment: {e}")
            print(f"   Using default experiment")
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run (e.g., "logistic-baseline-v1")
            tags: Additional tags for the run
        """
        # Add timestamp to run name if configured
        if self.config['run_naming']['include_timestamp']:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{run_name}-{timestamp}"
        
        self.current_run = mlflow.start_run(run_name=run_name)
        
        # Log default tags
        if tags:
            mlflow.set_tags(tags)
        
        print(f"\nStarted run: {run_name}")
        return self.current_run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not self.current_run:
            raise ValueError("No active run. Call start_run() first.")
        
        mlflow.log_params(params)
        print(f"   Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics to current run.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        if not self.current_run:
            raise ValueError("No active run. Call start_run() first.")
        
        mlflow.log_metrics(metrics)
        print(f"   Logged {len(metrics)} metrics")
    
    def log_confusion_matrix(self, y_true, y_pred, labels=None, save_path="confusion_matrix.png"):
        """
        Generate and log confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            save_path: Where to save the plot
        """
        if not self.current_run:
            raise ValueError("No active run. Call start_run() first.")
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels or ['No', 'Yes'],
                    yticklabels=labels or ['No', 'Yes'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save and log
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(save_path)
        print(f"   Logged confusion matrix")
        
        # Clean up
        Path(save_path).unlink()
    
    def log_roc_curve(self, y_true, y_pred_proba, save_path="roc_curve.png"):
        """
        Generate and log ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            save_path: Where to save the plot
        """
        if not self.current_run:
            raise ValueError("No active run. Call start_run() first.")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save and log
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(save_path)
        print(f"   Logged ROC curve")
        
        # Clean up
        Path(save_path).unlink()
    
    def log_feature_importance(self, feature_names, importance_values, 
                               top_n=20, save_path="feature_importance.png"):
        """
        Log feature importance plot and CSV.
        
        Args:
            feature_names: List of feature names
            importance_values: Importance scores for each feature
            top_n: Number of top features to plot
            save_path: Where to save the plot
        """
        if not self.current_run:
            raise ValueError("No active run. Call start_run() first.")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        # Save CSV
        csv_path = save_path.replace('.png', '.csv')
        importance_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save and log
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(save_path)
        print(f"   Logged feature importance")
        
        # Clean up
        Path(save_path).unlink()
        Path(csv_path).unlink()
    
    def log_model(self, model, artifact_path="model"):
        """
        Log trained model to MLflow.
        
        Args:
            model: Trained sklearn model
            artifact_path: Path within run artifacts
        """
        if not self.current_run:
            raise ValueError("No active run. Call start_run() first.")
        
        mlflow.sklearn.log_model(model, artifact_path)
        print(f"   Logged model to '{artifact_path}'")
    
    def end_run(self):
        """End current MLflow run."""
        if not self.current_run:
            print("No active run to end")
            return
        
        mlflow.end_run()
        print(f"Ended run\n")
        self.current_run = None


def evaluate_classification_model(
    model,
    X_test,
    y_test,
    feature_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of classification model.
    
    Args:
        model: Trained sklearn classifier
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names for importance logging
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    return metrics