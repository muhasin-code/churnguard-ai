"""
Train ensemble models combining best peroformers from experimentation.

- Your best single model: XGBoost Conservative (0.8231)
- Target ensemble ROC-AUC: 0.825-0.833
"""


import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.model_selection import learning_curve

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models.train_models import ModelExperiment


def main():
    """Train ensemble models."""
    print("\n" + "=" * 70)
    print("ChurnGuard AI - Ensemble Training")
    print("=" * 70 + "\n")

    # Inintialize experiment 
    experiment = ModelExperiment()
    experiment.load_data()

    # ===========================================
    # Define Base Models 
    # ===========================================

    print("Creating base models based on Phase 1 & 2 results...\n")

    # Random Forest (best was rf-forest: 0.8159)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    # XGBoost (best was xgb-conservative: 0.8231)
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Calculate scale_pos_weight
    scale_pos_weight = (experiment.y_train == 0).sum() / (experiment.y_train == 1).sum()
    xgb_model.set_params(scale_pos_weight=scale_pos_weight)

    # LightGBM (best was lgbm-tuned: 0.8212)
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=-1
    )

    # ===========================================
    # Ensemble 1: Voting Classifier (Soft)
    # ===========================================

    print("Training Voting Ensemble (Soft Voting)...\n")

    voting_models = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ]

    experiment.train_ensemble(
        models=voting_models,
        run_name="voting-soft-ensemble",
        voting='soft'
    )

    # ===========================================
    # Ensemble 2: Stacking Classifier
    # ===========================================
    print("Training Stacking Ensemble...\n")

    # Stacking uses meta-model to combine base models
    from src.models.mlflow_utils import MLflowExperimentTracker, evaluate_classification_model

    tracker = MLflowExperimentTracker()

    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=voting_models,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
        n_jobs=-1
    )

    tracker.start_run(
        run_name="stacking-ensemble",
        tags={'model_type': 'stacking', 'stage': 'experimentation'}
    )

    # Log parameters
    tracker.log_params({
        'base_models': ['RandomForest', 'XGBoost', 'LightGBM'],
        'meta_model': 'LogisticRegression',
        'cv_folds': 5
    })

    # Train
    print("Training stacking ensemble...")
    stacking.fit(experiment.X_train, experiment.y_train)

    # Evaluate
    print("Evaluating stacking ensemble...")
    metrics = evaluate_classification_model(
        stacking, experiment.X_test, experiment.y_test, experiment.feature_names
    )

    # Log metrics
    tracker.log_metrics(metrics)

    # Generate visualizations
    y_pred = stacking.predict(experiment.X_test)
    y_pred_proba = stacking.predict_proba(experiment.X_test)[:, 1]

    tracker.log_confusion_matrix(experiment.y_test, y_pred)
    tracker.log_roc_curve(experiment.y_test, y_pred_proba)

    # Log model
    tracker.log_model(stacking)

    # Store results
    experiment.results.append({
        'run_name': 'stacking-ensemble',
        'model_type': 'Stacking',
        'roc_auc': metrics['roc_auc'],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    })
    
    tracker.end_run()
    
    print(f"Stacking ensemble complete!")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}\n")

    # ===========================================
    # Final Comparison
    # ===========================================

    experiment.print_comparison()

    print("\n" + "=" * 70)
    print("Ensemble Training Complete!")
    print("=" * 70)

    # Adjusted expectations message
    print("\nPerformance Analysis:")
    print(f"   Baseline (Logistic Regression): 0.8003 ROC-AUC")
    print(f"   Best Single Model (XGBoost):    0.8231 ROC-AUC (+2.3%)")
    print(f"   Expected Ensemble Range:        0.825-0.833 ROC-AUC")
    
    if len(experiment.results) > 0:
        best_ensemble = max(experiment.results, key=lambda x: x['roc_auc'])
        improvement = best_ensemble['roc_auc'] - 0.8231
        print(f"   Best Ensemble Achieved:         {best_ensemble['roc_auc']:.4f} ROC-AUC ({improvement:+.4f})")
    
    print("\nNext Steps:")
    print("   1. Review all models in MLflow UI")
    print("   2. Champion model: XGBoost Conservative (0.8231 ROC-AUC)")
    print("      - Best balance of performance and simplicity")
    print("      - Ensembles add minimal gain (<1%) for 3x complexity")
    print("   3. Proceed to Milestone 2.3 for detailed evaluation\n")


if __name__ == "__main__":
    main()