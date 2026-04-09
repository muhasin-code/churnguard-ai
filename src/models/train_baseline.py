"""
Train baseline Logistic Regression model for churn prediction.

This establishes the minimum acceptable performance bar for ChurnGuard AI.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.models.mlflow_utils import MLflowExperimentTracker, evaluate_classification_model


class BaselineChurnModel:
    """
    Baseline Logistic Regression model for churn prediction.
    
    This serves as the performance benchmark for more complex models.
    
    Example:
        model = BaselineChurnModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(self, **kwargs):
        """
        Initialize baseline model.
        
        Args:
            **kwargs: Hyperparameters for LogisticRegression
                     (C, penalty, solver, max_iter, etc.)
        """
        # Default hyperparameters
        default_params = {
            'C': 1.0,                    # Inverse of regularization strength
            'penalty': 'l2',             # L2 regularization (Ridge)
            'solver': 'lbfgs',           # Optimization algorithm
            'max_iter': 1000,            # Maximum iterations
            'random_state': 42,          # For reproducibility
            'class_weight': 'balanced',  # Handle class imbalance
            'n_jobs': -1                 # Use all CPU cores
        }
        
        # Override defaults with provided kwargs
        self.params = {**default_params, **kwargs}
        
        # Initialize model
        self.model = LogisticRegression(**self.params)
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features (DataFrame or ndarray)
            y_train: Training target (Series or ndarray)
            feature_names: List of feature names (for importance logging)
        """
        print("  Training Logistic Regression baseline...")
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training set performance
        train_score = self.model.score(X_train, y_train)
        print(f"   Training accuracy: {train_score:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\nEvaluating on test set...")
        
        metrics = evaluate_classification_model(
            self.model, X_test, y_test, self.feature_names
        )
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance from model coefficients.
        
        For logistic regression, we use absolute value of coefficients
        as a proxy for feature importance.
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        # Get coefficients (weights)
        coefficients = self.model.coef_[0]
        
        # Use absolute value as importance
        importance = np.abs(coefficients)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        joblib.dump(self.model, filepath)
        print(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load trained model from disk."""
        model_instance = cls()
        model_instance.model = joblib.load(filepath)
        model_instance.is_trained = True
        print(f"Loaded model from {filepath}")
        return model_instance


def main():
    """
    Main training pipeline for baseline model.
    
    This function:
    1. Loads processed features
    2. Trains logistic regression
    3. Evaluates on test set
    4. Logs everything to MLflow
    """
    
    print("=" * 70)
    print("ChurnGuard AI - Baseline Model Training")
    print("=" * 70)
    
    # ==========================================
    # 1. Load Data
    # ==========================================
    
    print("\nLoading processed features...")
    
    train_path = "data/processed/features_v1_train.csv"
    test_path = "data/processed/features_v1_test.csv"
    
    if not Path(train_path).exists():
        print(f"Error: Training data not found at {train_path}")
        print("   Run 'python scripts/engineer_features.py' first")
        sys.exit(1)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Train: {train_df.shape}")
    print(f"   Test:  {test_df.shape}")
    
    # Separate features and target
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    feature_names = list(X_train.columns)
    
    print(f"   Features: {len(feature_names)}")
    print(f"   Train target distribution: {y_train.value_counts().to_dict()}")
    
    # ==========================================
    # 2. Initialize MLflow Tracker
    # ==========================================
    
    print("\nSetting up MLflow tracking...")
    tracker = MLflowExperimentTracker()
    
    # ==========================================
    # 3. Define Hyperparameters
    # ==========================================
    
    hyperparameters = {
        'C': 1.0,                    # Regularization strength (inverse)
        'penalty': 'l2',             # L2 regularization
        'solver': 'lbfgs',           # Optimizer
        'max_iter': 1000,            # Max iterations
        'class_weight': 'balanced',  # Handle imbalance
        'random_state': 42
    }
    
    print(f"\nHyperparameters:")
    for param, value in hyperparameters.items():
        print(f"   {param}: {value}")
    
    # ==========================================
    # 4. Start MLflow Run
    # ==========================================
    
    tracker.start_run(
        run_name="logistic-baseline",
        tags={
            'model_type': 'logistic_regression',
            'stage': 'baseline',
            'data_version': 'v1'
        }
    )
    
    # Log hyperparameters
    tracker.log_params(hyperparameters)
    
    # Log dataset info
    tracker.log_params({
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_names),
        'train_churn_rate': (y_train == 1).mean(),
        'test_churn_rate': (y_test == 1).mean()
    })
    
    # ==========================================
    # 5. Train Model
    # ==========================================
    
    model = BaselineChurnModel(**hyperparameters)
    model.train(X_train, y_train, feature_names=feature_names)
    
    # ==========================================
    # 6. Evaluate Model
    # ==========================================
    
    metrics = model.evaluate(X_test, y_test)
    
    # Log metrics to MLflow
    tracker.log_metrics(metrics)
    
    # Print summary
    print(f"\nTest Set Performance:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:    {metrics['pr_auc']:.4f}")
    
    # ==========================================
    # 7. Generate and Log Visualizations
    # ==========================================
    
    print(f"\nGenerating visualizations...")
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    tracker.log_confusion_matrix(
        y_test, y_pred,
        labels=['No Churn', 'Churn']
    )
    
    # ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    tracker.log_roc_curve(y_test, y_pred_proba)
    
    # Feature importance
    importance_df = model.get_feature_importance()
    tracker.log_feature_importance(
        importance_df['feature'].values,
        importance_df['importance'].values,
        top_n=20
    )
    
    print(f"\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:30s} | Coef: {row['coefficient']:>8.4f} | Importance: {row['importance']:>8.4f}")
    
    # ==========================================
    # 8. Log Model to MLflow
    # ==========================================
    
    print(f"\nLogging model to MLflow...")
    tracker.log_model(model.model, artifact_path="model")
    
    # ==========================================
    # 9. Save Model Locally (Optional)
    # ==========================================
    
    model_save_path = "models/baseline_logistic_regression.pkl"
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    
    # ==========================================
    # 10. End MLflow Run
    # ==========================================
    
    tracker.end_run()
    
    # ==========================================
    # 11. Summary
    # ==========================================
    
    print("=" * 70)
    print("Baseline Model Training Complete!")
    print("=" * 70)
    print(f"\nKey Metrics:")
    print(f"   Primary (ROC-AUC): {metrics['roc_auc']:.4f}")
    print(f"   Accuracy:          {metrics['accuracy']:.4f}")
    print(f"   Precision:         {metrics['precision']:.4f}")
    print(f"   Recall:            {metrics['recall']:.4f}")
    
    print(f"\nArtifacts:")
    print(f"   Model: {model_save_path}")
    print(f"   MLflow UI: http://localhost:5000")
    
    print(f"\nNext Steps:")
    print(f"   1. Review results in MLflow UI")
    print(f"   2. Analyze feature importance")
    print(f"   3. Compare with advanced models (XGBoost, Random Forest)")
    print(f"   4. Document baseline performance in project README")
    

if __name__ == "__main__":
    main()