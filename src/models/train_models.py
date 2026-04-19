"""
Train and compare multiple models for churn prediction.

This script trains Random Forest, XGBoost, LightGBM, and ensemble models,
logging all experiments to MLflow for systematic comparison.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models.mlflow_utils import MLflowExperimentTracker, evaluate_classification_model


class ModelExperiment:
    """
    Manages training and evaluation of multiple model types.
    
    Example:
        experiment = ModelExperiment()
        experiment.load_data()
        experiment.train_random_forest(n_estimators=100)
        experiment.train_xgboost(learning_rate=0.1)
        experiment.compare_models()
    """
    
    def __init__(self):
        """Initialize experiment manager."""
        self.tracker = MLflowExperimentTracker()
        
        # Data containers
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
        # Results tracking
        self.results = []
    
    def load_data(self):
        """Load processed training and test data."""
        print("Loading processed features...")
        
        train_path = "data/processed/features_v1_train.csv"
        test_path = "data/processed/features_v1_test.csv"
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Separate features and target
        self.X_train = train_df.drop(columns=['Churn'])
        self.y_train = train_df['Churn']
        
        self.X_test = test_df.drop(columns=['Churn'])
        self.y_test = test_df['Churn']
        
        self.feature_names = list(self.X_train.columns)
        
        print(f"   Train: {self.X_train.shape}")
        print(f"   Test:  {self.X_test.shape}")
        print(f"   Features: {len(self.feature_names)}\n")
    
    def train_random_forest(
        self,
        run_name: str = "random-forest-default",
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        **kwargs
    ):
        """
        Train Random Forest classifier.
        
        Args:
            run_name: MLflow run name
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Features to consider at each split
            **kwargs: Additional RandomForest parameters
        """
        print("=" * 70)
        print(f"Training Random Forest: {run_name}")
        print("=" * 70)
        
        # Hyperparameters
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced',
            **kwargs
        }
        
        # Start MLflow run
        self.tracker.start_run(
            run_name=run_name,
            tags={'model_type': 'random_forest', 'stage': 'experimentation'}
        )
        
        # Log parameters
        self.tracker.log_params(params)
        self.tracker.log_params({
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'n_features': len(self.feature_names)
        })
        
        # Train model
        print("Training model...")
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        print("Evaluating model...")
        metrics = evaluate_classification_model(
            model, self.X_test, self.y_test, self.feature_names
        )
        
        # Log metrics
        self.tracker.log_metrics(metrics)
        
        # Generate visualizations
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        self.tracker.log_confusion_matrix(self.y_test, y_pred)
        self.tracker.log_roc_curve(self.y_test, y_pred_proba)
        self.tracker.log_feature_importance(
            self.feature_names,
            model.feature_importances_
        )
        
        # Log model
        self.tracker.log_model(model)
        
        # Store results
        self.results.append({
            'run_name': run_name,
            'model_type': 'RandomForest',
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        
        # End run
        self.tracker.end_run()
        
        print(f"Random Forest training complete!")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}\n")
        
        return model, metrics
    
    def train_xgboost(
        self,
        run_name: str = "xgboost-default",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: int = 1,
        gamma: float = 0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        **kwargs
    ):
        """
        Train XGBoost classifier.
        
        Args:
            run_name: MLflow run name
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage (0.01-0.3)
            max_depth: Maximum tree depth
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction to split
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            **kwargs: Additional XGBoost parameters
        """
        print("=" * 70)
        print(f"Training XGBoost: {run_name}")
        print("=" * 70)
        
        # Hyperparameters
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            **kwargs
        }
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        # Start MLflow run
        self.tracker.start_run(
            run_name=run_name,
            tags={'model_type': 'xgboost', 'stage': 'experimentation'}
        )
        
        # Log parameters
        self.tracker.log_params(params)
        self.tracker.log_params({
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'n_features': len(self.feature_names)
        })
        
        # Train model
        print("Training model...")
        model = XGBClassifier(**params)
        
        # Fit with evaluation set for early stopping monitoring
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        # Evaluate
        print("Evaluating model...")
        metrics = evaluate_classification_model(
            model, self.X_test, self.y_test, self.feature_names
        )
        
        # Log metrics
        self.tracker.log_metrics(metrics)
        
        # Generate visualizations
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        self.tracker.log_confusion_matrix(self.y_test, y_pred)
        self.tracker.log_roc_curve(self.y_test, y_pred_proba)
        self.tracker.log_feature_importance(
            self.feature_names,
            model.feature_importances_
        )
        
        # Log model
        self.tracker.log_model(model)
        
        # Store results
        self.results.append({
            'run_name': run_name,
            'model_type': 'XGBoost',
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        
        # End run
        self.tracker.end_run()
        
        print(f"XGBoost training complete!")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}\n")
        
        return model, metrics
    
    def train_lightgbm(
        self,
        run_name: str = "lightgbm-default",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        **kwargs
    ):
        """
        Train LightGBM classifier.
        
        Args:
            run_name: MLflow run name
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage
            max_depth: Maximum tree depth (-1 = no limit)
            num_leaves: Maximum leaves per tree
            min_child_samples: Minimum samples in leaf
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            **kwargs: Additional LightGBM parameters
        """
        print("=" * 70)
        print(f"Training LightGBM: {run_name}")
        print("=" * 70)
        
        # Hyperparameters
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced',
            'verbose': -1,
            **kwargs
        }
        
        # Start MLflow run
        self.tracker.start_run(
            run_name=run_name,
            tags={'model_type': 'lightgbm', 'stage': 'experimentation'}
        )
        
        # Log parameters
        self.tracker.log_params(params)
        self.tracker.log_params({
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'n_features': len(self.feature_names)
        })
        
        # Train model
        print("Training model...")
        model = LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        print("Evaluating model...")
        metrics = evaluate_classification_model(
            model, self.X_test, self.y_test, self.feature_names
        )
        
        # Log metrics
        self.tracker.log_metrics(metrics)
        
        # Generate visualizations
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        self.tracker.log_confusion_matrix(self.y_test, y_pred)
        self.tracker.log_roc_curve(self.y_test, y_pred_proba)
        self.tracker.log_feature_importance(
            self.feature_names,
            model.feature_importances_
        )
        
        # Log model
        self.tracker.log_model(model)
        
        # Store results
        self.results.append({
            'run_name': run_name,
            'model_type': 'LightGBM',
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        
        # End run
        self.tracker.end_run()
        
        print(f"LightGBM training complete!")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}\n")
        
        return model, metrics
    
    def train_ensemble(
        self,
        models: list,
        run_name: str = "voting-ensemble",
        voting: str = 'soft'
    ):
        """
        Train voting ensemble combining multiple models.
        
        Args:
            models: List of (name, model) tuples
            run_name: MLflow run name
            voting: 'hard' (majority vote) or 'soft' (average probabilities)
        """
        print("=" * 70)
        print(f"Training Ensemble: {run_name}")
        print("=" * 70)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting=voting,
            n_jobs=-1
        )
        
        # Start MLflow run
        self.tracker.start_run(
            run_name=run_name,
            tags={'model_type': 'ensemble', 'stage': 'experimentation'}
        )
        
        # Log parameters
        self.tracker.log_params({
            'voting': voting,
            'n_models': len(models),
            'models': [name for name, _ in models]
        })
        
        # Train
        print("Training ensemble...")
        ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate
        print("Evaluating ensemble...")
        metrics = evaluate_classification_model(
            ensemble, self.X_test, self.y_test, self.feature_names
        )
        
        # Log metrics
        self.tracker.log_metrics(metrics)
        
        # Generate visualizations
        y_pred = ensemble.predict(self.X_test)
        y_pred_proba = ensemble.predict_proba(self.X_test)[:, 1]
        
        self.tracker.log_confusion_matrix(self.y_test, y_pred)
        self.tracker.log_roc_curve(self.y_test, y_pred_proba)
        
        # Log model
        self.tracker.log_model(ensemble)
        
        # Store results
        self.results.append({
            'run_name': run_name,
            'model_type': 'Ensemble',
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        
        # End run
        self.tracker.end_run()
        
        print(f"Ensemble training complete!")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}\n")
        
        return ensemble, metrics
    
    def print_comparison(self):
        """Print comparison table of all trained models."""
        if not self.results:
            print("No models trained yet!")
            return
        
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        print("\n" + "=" * 90)
        print("MODEL COMPARISON (sorted by ROC-AUC)")
        print("=" * 90)
        print(f"{'Rank':<6} {'Model Type':<20} {'Run Name':<30} {'ROC-AUC':<10} {'Accuracy':<10}")
        print("-" * 90)
        
        for idx, row in enumerate(results_df.itertuples(), 1):
            print(f"{idx:<6} {row.model_type:<20} {row.run_name:<30} {row.roc_auc:<10.4f} {row.accuracy:<10.4f}")
        
        print("=" * 90)
        
        # Highlight best model
        best = results_df.iloc[0]
        print(f"\nBest Model: {best['model_type']} ({best['run_name']})")
        print(f"   ROC-AUC: {best['roc_auc']:.4f}")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"   Precision: {best['precision']:.4f}")
        print(f"   Recall: {best['recall']:.4f}\n")


def main():
    """
    Main experimentation pipeline.
    
    Trains multiple models systematically and compares performance.
    """
    print("\n" + "=" * 70)
    print("ChurnGuard AI - Model Experimentation")
    print("=" * 70 + "\n")
    
    # Initialize experiment
    experiment = ModelExperiment()
    experiment.load_data()
    
    # ======================================
    # PHASE 1: Default Configurations
    # ======================================
    
    print("\nPHASE 1: Default Configurations\n")
    
    # Random Forest - default
    experiment.train_random_forest(
        run_name="rf-default",
        n_estimators=100
    )
    
    # XGBoost - default
    experiment.train_xgboost(
        run_name="xgb-default",
        n_estimators=100,
        learning_rate=0.1
    )
    
    # LightGBM - default
    experiment.train_lightgbm(
        run_name="lgbm-default",
        n_estimators=100,
        learning_rate=0.1
    )
    
    # ======================================
    # PHASE 2: Tuned Configurations
    # ======================================
    
    print("\nPHASE 2: Tuned Configurations\n")
    
    # Random Forest - deeper trees
    experiment.train_random_forest(
        run_name="rf-deep",
        n_estimators=200,
        max_depth=20,
        min_samples_split=5
    )
    
    # Random Forest - more trees
    experiment.train_random_forest(
        run_name="rf-forest",
        n_estimators=300,
        max_depth=15
    )
    
    # XGBoost - conservative (less overfitting)
    experiment.train_xgboost(
        run_name="xgb-conservative",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    # XGBoost - aggressive (more capacity)
    experiment.train_xgboost(
        run_name="xgb-aggressive",
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9
    )
    
    # LightGBM - tuned
    experiment.train_lightgbm(
        run_name="lgbm-tuned",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    # Print comparison so far
    experiment.print_comparison()
    
    print("\nReview results in MLflow UI: http://localhost:5000")
    print("   Compare models before proceeding to ensembles.\n")


if __name__ == "__main__":
    main()