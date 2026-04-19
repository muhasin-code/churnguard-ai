"""
Model evaluation utilities for ChurnGuard AI.

Provides comprehensive evaluation tools including SHAP explanations,
calibration analysis, error analysis, and threshold optimization.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import shap

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve


class ModelEvaluator:
    """
    Comprehensive model evaluation and interpretation.
    
    Example:
        evaluator = ModelEvaluator(model, X_test, y_test, feature_names)
        evaluator.generate_shap_explanations()
        evaluator.plot_confusion_matrix()
        evaluator.analyze_errors()
    """
    
    def __init__(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model (must have predict_proba method)
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            output_dir: Directory to save outputs
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate predictions once
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Will be populated later
        self.shap_values = None
        self.shap_explainer = None
        self.X_shap = None
        self.shap_sample_indices = None
    
    # =========================================================================
    # SHAP Explanations
    # =========================================================================
    
    def generate_shap_explanations(self, sample_size: Optional[int] = None):
        """
        Generate SHAP values for test set.
        
        Uses TreeExplainer for tree-based models (fast and exact).
        
        Args:
            sample_size: If provided, compute SHAP for random sample only
                        (useful for large datasets). None = use all data.
        """
        print("Generating SHAP explanations...")
        print(f"   Model type: {type(self.model).__name__}")
        
        # Sample data if requested
        if sample_size and sample_size < len(self.X_test):
            print(f"   Using random sample of {sample_size} observations")
            self.shap_sample_indices = np.random.choice(
                len(self.X_test), 
                size=sample_size, 
                replace=False
            )
            X_sample = self.X_test.iloc[self.shap_sample_indices]
        else:
            self.shap_sample_indices = np.arange(len(self.X_test))
            X_sample = self.X_test
            print(f"   Computing for all {len(X_sample)} test observations")
        
        self.X_shap = X_sample
        
        # Create explainer (TreeExplainer for XGBoost/LightGBM/RF)
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("   Using TreeExplainer (fast, exact for tree models)")
        except:
            # Fallback to KernelExplainer (slower, model-agnostic)
            print("   Falling back to KernelExplainer (slower)")
            background = shap.sample(self.X_test, 100)  # Background dataset
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
        
        # Compute SHAP values
        self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Handle multi-output (for binary classification, take positive class)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class
        
        print(f"SHAP values computed!")
        print(f"   Shape: {self.shap_values.shape}")
        
        return self.shap_values
    
    def plot_shap_summary(self, max_display: int = 20):
        """
        Generate SHAP summary plot (global feature importance).
        
        Args:
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            raise ValueError("Generate SHAP values first with generate_shap_explanations()")
        
        print(f"Creating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_shap,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        save_path = self.output_dir / "shap_summary_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
    
    def plot_shap_waterfall(self, customer_index: int):
        """
        Generate SHAP waterfall plot for single prediction.
        
        Args:
            customer_index: Index of customer in test set
        """
        if self.shap_values is None:
            raise ValueError("Generate SHAP values first with generate_shap_explanations()")
        
        # Translate global test index to loacl SHAP sample index
        local_indices = np.where(self.shap_sample_indices == customer_index)[0]
        if len(local_indices) == 0:
            print(f"   Customer {customer_index} not in SHAP sample, skipping.")
            return
        local_index = local_indices[0]
        
        print(f"Creating SHAP waterfall for customer {customer_index}...")
        
        # Create Explanation object (required for waterfall plot)
        explanation = shap.Explanation(
            values=self.shap_values[local_index],
            base_values=self.shap_explainer.expected_value,
            data=self.X_shap.iloc[local_index].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        
        save_path = self.output_dir / f"shap_waterfall_customer_{customer_index}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
    
    def plot_shap_dependence(self, feature: str, interaction_feature: str = None):
        """
        Generate SHAP dependence plot for a feature.
        
        Args:
            feature: Feature to plot
            interaction_feature: Feature to color by (auto-detected if None)
        """
        if self.shap_values is None:
            raise ValueError("Generate SHAP values first with generate_shap_explanations()")
        
        print(f"Creating SHAP dependence plot for {feature}...")
        
        # Get feature index
        feature_idx = self.feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.X_shap,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        
        save_path = self.output_dir / f"shap_dependence_{feature}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
    
    def get_top_shap_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N features by mean absolute SHAP value.
        
        Args:
            n: Number of top features
            
        Returns:
            DataFrame with features and importance scores
        """
        if self.shap_values is None:
            raise ValueError("Generate SHAP values first with generate_shap_explanations()")
        
        # Mean absolute SHAP value = global importance
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance_df.head(n)
    
    # =========================================================================
    # Confusion Matrix & Classification Report
    # =========================================================================
    
    def plot_confusion_matrix(self, normalize: bool = False):
        """
        Plot confusion matrix.
        
        Args:
            normalize: If True, show proportions instead of counts
        """
        print("Creating confusion matrix...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        
        filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
    
    def print_classification_report(self):
        """Print detailed classification report."""
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=['No Churn', 'Churn'],
            digits=4
        ))
    
    # =========================================================================
    # ROC & Precision-Recall Curves
    # =========================================================================
    
    def plot_roc_curve(self):
        """Plot ROC curve with AUC."""
        print("Creating ROC curve...")
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})'
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color='navy',
            lw=2,
            linestyle='--',
            label='Random Classifier'
        )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
        print(f"   AUC: {roc_auc:.4f}")
    
    def plot_precision_recall_curve(self):
        """Plot Precision-Recall curve."""
        print("Creating Precision-Recall curve...")
        
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba
        )
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color='darkorange',
            lw=2,
            label=f'PR curve (AUC = {pr_auc:.4f})'
        )
        
        # Baseline (proportion of positive class)
        baseline = self.y_test.mean()
        plt.plot(
            [0, 1],
            [baseline, baseline],
            color='navy',
            lw=2,
            linestyle='--', label=f'Baseline ({baseline:.2f})'
        )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "precision_recall_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
        print(f"   PR-AUC: {pr_auc:.4f}")
    
    # =========================================================================
    # Calibration Analysis
    # =========================================================================
    
    def plot_calibration_curve(self, n_bins: int = 10):
        """
        Plot calibration curve to assess probability calibration.
        
        Args:
            n_bins: Number of bins for calibration
        """
        print("Creating calibration curve...")
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test,
            self.y_pred_proba,
            n_bins=n_bins,
            strategy='uniform'
        )
        
        # Calculate Brier score (lower is better)
        brier_score = brier_score_loss(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        
        # Perfect calibration line
        plt.plot(
            [0, 1],
            [0, 1],
            'k--',
            lw=2,
            label='Perfect Calibration'
        )
        
        # Model calibration
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            's-',
            label=f'Model (Brier={brier_score:.4f})'
        )
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "calibration_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
        print(f"   Brier Score: {brier_score:.4f} (lower is better)")
        
        # Interpretation
        if brier_score < 0.1:
            print("   Excellent calibration")
        elif brier_score < 0.2:
            print("   Good calibration")
        else:
            print("   Poor calibration - consider recalibration")
    
    # =========================================================================
    # Error Analysis
    # =========================================================================
    
    def analyze_errors(self) -> pd.DataFrame:
        """
        Analyze prediction errors to find patterns.
        
        Returns:
            DataFrame with error analysis
        """
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)
        
        # Create error dataframe
        error_df = self.X_test.copy()
        error_df['true_label'] = self.y_test
        error_df['predicted_label'] = self.y_pred
        error_df['predicted_proba'] = self.y_pred_proba
        error_df['correct'] = (self.y_test == self.y_pred)
        
        # Categorize errors
        error_df['error_type'] = 'Correct'
        error_df.loc[
            (error_df['true_label'] == 0) & (error_df['predicted_label'] == 1),
            'error_type'
        ] = 'False Positive'
        error_df.loc[
            (error_df['true_label'] == 1) & (error_df['predicted_label'] == 0),
            'error_type'
        ] = 'False Negative'
        
        # Summary statistics
        print(f"\nOverall Accuracy: {error_df['correct'].mean():.2%}")
        print(f"\nError Breakdown:")
        print(error_df['error_type'].value_counts())
        
        # Analyze False Positives
        fp_df = error_df[error_df['error_type'] == 'False Positive']
        if len(fp_df) > 0:
            print(f"\nFalse Positives ({len(fp_df)} cases):")
            print("   Customers incorrectly flagged as churners")
            print(f"   Mean predicted probability: {fp_df['predicted_proba'].mean():.2%}")
            
            # Top features for FP
            print("\n   Characteristics (mean values):")
            for col in ['Tenure', 'MonthlyCharges', 'TotalCharges']:
                if col in fp_df.columns:
                    print(f"   - {col}: {fp_df[col].mean():.2f}")
        
        # Analyze False Negatives
        fn_df = error_df[error_df['error_type'] == 'False Negative']
        if len(fn_df) > 0:
            print(f"\nFalse Negatives ({len(fn_df)} cases):")
            print("   Churners we missed")
            print(f"   Mean predicted probability: {fn_df['predicted_proba'].mean():.2%}")
            
            # Top features for FN
            print("\n   Characteristics (mean values):")
            for col in ['Tenure', 'MonthlyCharges', 'TotalCharges']:
                if col in fn_df.columns:
                    print(f"   - {col}: {fn_df[col].mean():.2f}")
        
        # Save error analysis
        save_path = self.output_dir / "error_analysis.csv"
        error_df.to_csv(save_path, index=False)
        print(f"\nSaved error analysis to {save_path}")
        
        return error_df
    
    def plot_error_distribution(self):
        """Plot distribution of prediction probabilities by error type."""
        print("Creating error distribution plot...")
        
        # Create error dataframe
        error_df = self.X_test.copy()
        error_df['true_label'] = self.y_test
        error_df['predicted_proba'] = self.y_pred_proba
        error_df['predicted_label'] = self.y_pred
        
        # Categorize
        error_df['error_type'] = 'True Negative'
        error_df.loc[
            (error_df['true_label'] == 1) & (error_df['predicted_label'] == 1),
            'error_type'
        ] = 'True Positive'
        error_df.loc[
            (error_df['true_label'] == 0) & (error_df['predicted_label'] == 1),
            'error_type'
        ] = 'False Positive'
        error_df.loc[
            (error_df['true_label'] == 1) & (error_df['predicted_label'] == 0),
            'error_type'
        ] = 'False Negative'
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, error_type in enumerate(['True Negative', 'True Positive',
                                          'False Positive', 'False Negative']):
            subset = error_df[error_df['error_type'] == error_type]
            
            axes[idx].hist(
                subset['predicted_proba'],
                bins=30,
                edgecolor='black',
                alpha=0.7
            )
            axes[idx].axvline(
                0.5,
                color='red',
                linestyle='--',
                lw=2,
                label='Threshold (0.5)'
            )
            axes[idx].set_xlabel('Predicted Probability')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'{error_type} (n={len(subset)})')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "error_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to {save_path}")
    
    # =========================================================================
    # Threshold Optimization
    # =========================================================================
    
    def find_optimal_threshold(
        self,
        cost_fp: float = 50,
        cost_fn: float = 600
    ) -> Tuple[float, Dict]:
        """
        Find optimal classification threshold based on business costs.
        
        Args:
            cost_fp: Cost of false positive (wasted retention campaign)
            cost_fn: Cost of false negative (lost customer)
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        print(f"\nFinding optimal threshold...")
        print(f"   Cost of False Positive: ${cost_fp}")
        print(f"   Cost of False Negative: ${cost_fn}")
        
        # Try different thresholds (search from 0.01 to avoid floor effect)
        thresholds = np.linspace(0.01, 0.9, 90)
        costs = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(self.y_test, y_pred_thresh)
            
            tn, fp, fn, tp = cm.ravel()
            
            # Total cost = (FP × cost_fp) + (FN × cost_fn)
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            costs.append(total_cost)
        
        # Find minimum cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_cost = costs[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (self.y_pred_proba >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(self.y_test, y_pred_optimal)
        tn, fp, fn, tp = cm_optimal.ravel()
        
        optimal_metrics = {
            'threshold':        float(optimal_threshold),
            'total_cost':       float(optimal_cost),
            'true_negatives':   int(tn),
            'false_positives':  int(fp),
            'false_negatives':  int(fn),
            'true_positives':   int(tp),
            'precision':        float(tp / (tp + fp) if (tp + fp) > 0 else 0),
            'recall':           float(tp / (tp + fn) if (tp + fn) > 0 else 0)
        }
        
        print(f"\nOptimal threshold: {optimal_threshold:.3f}")
        print(f"   Total cost: ${optimal_cost:,.0f}")
        print(f"   Precision: {optimal_metrics['precision']:.2%}")
        print(f"   Recall: {optimal_metrics['recall']:.2%}")
        print(f"   False Positives: {fp}")
        print(f"   False Negatives: {fn}")
        
        # Plot cost vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, costs, 'b-', lw=2)
        plt.axvline(optimal_threshold, color='r', linestyle='--', lw=2,
                   label=f'Optimal ({optimal_threshold:.3f})')
        plt.axvline(0.5, color='gray', linestyle=':', lw=2,
                   label='Default (0.5)')
        plt.xlabel('Classification Threshold')
        plt.ylabel('Total Cost ($)')
        plt.title('Cost vs Classification Threshold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "threshold_optimization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved threshold plot to {save_path}")
        
        return optimal_threshold, optimal_metrics