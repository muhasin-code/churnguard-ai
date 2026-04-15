"""
Comprehensive evalutaion of chanmpion model.

This script performs depp evaluation including SHAP analysis, error analysis, calibration assessment, and threshold optimization.
"""


import pandas as pd
import numpy as np
import sys
from pathlib import Path
import mlflow
import joblib

# Add parent to your path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluation_utils import ModelEvaluator


def load_champion_model():
    """Load champion model from MLflow or local file."""

    # Try loading from local file first -> faster
    model_path = Path("models/baseline_logistic_regression.pkl")

    # Check if XGBoost model exists
    xgb_path = Path("models/xgboost_conservative.pkl")

    if xgb_path.exists():
        print(f"Loading champion model from {xgb_path}")
        model = joblib.load(xgb_path)
        return model
    elif model_path.exists():
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model
    else:
        print("Champion model not found locally")
        print("   Expected path: models/xgboost_conservative.pkl")
        print("\n   To save champion model:")
        print("   1. Train XGBoost: python src/models/train_models.py")
        print("   2. Save model manually or copy from MLflow artifacts")
        sys.exit(1)


def main():
    """Main evaluation pipeline."""

    print("=" * 70)
    print("ChurnGuard AI - Champion Model Evaluation")
    print("=" * 70)

    # ==========================================================
    # 1. Load Data
    # ==========================================================
    print("\nLoading processed features...")

    train_df = pd.read_csv("data/processed/features_v1_train.csv")
    test_df = pd.read_csv("data/processed/features_v1_test.csv")

    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    feature_names = list(X_test.columns)

    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {len(feature_names)}")

    # ==========================================================
    # 2. Load Champion Model
    # ==========================================================
    print("\nLoading champion model...")
    model = load_champion_model()

    print(f"   Model type: {type(model).__name__}")

    # ==========================================================
    # 3. Initialize Evaluator
    # ==========================================================
    print("\nInitializing Evaluator...")

    evaluator = ModelEvaluator(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir="evaluation_results"
    )

    print("   Output directory: evaluation_results/")

    # ==========================================================
    # 4. Basic Metrics
    # ==========================================================
    print("\n" + "=" * 70)
    print("STEP 1: Basic Performance Metrics")
    print("=" * 70)

    evaluator.print_classification_report()

    # ==========================================================
    # 5. Confusion Matrix
    # ==========================================================
    print("\n" + "=" * 70)
    print("STEP 2: Confusion Matrix Analysis")
    print("=" * 70)

    evaluator.plot_confusion_matrix(normalize=False)
    evaluator.plot_confusion_matrix(normalize=True)

    # ==========================================================
    # 6. ROC & Precision-Recall Curve
    # ==========================================================
    print("\n" + "=" * 70)
    print("STEP 3: ROC and Precision-Recall Curve")
    print("=" * 70)

    evaluator.plot_roc_curve()
    evaluator.plot_precision_recall_curve()

    # ==========================================================
    # 7. Calibration Analysis
    # ==========================================================
    print("\n" + "=" * 70)
    print("STEP 4: Probability Calibration")
    print("=" * 70)

    evaluator.plot_calibration_curve(n_bins=10)

    # ==========================================================
    # 8. SHAP Explanations
    # ==========================================================
    print("\n" + "=" * 70)
    print("STEP 5: SHAP Explanations (Model Interpretaility)")
    print("=" * 70)

    # Generate SHAP values (use sample for speed)
    # For 10K test set, computing all SHAP values takes ~2-3 minutes
    # Using sampleof 1,000 takes ~20 seconds
    evaluator.generate_shap_explanations(sample_size=1000)

    # Global importance
    print("\nGlobal feature importance (SHAP):")
    top_features = evaluator.get_top_shap_features(n=10)
    print(top_features.to_string(index=False))

    # Summary plot
    evaluator.plot_shap_summary(max_display=20)

    # Waterfall plots for example customers
    print("\nIndividual Explanations (Waterfall Plots):")

    # Find interesting cases
    shap_probas = evaluator.y_pred_proba[evaluator.shap_sample_indices]
    high_risk_idx = evaluator.shap_sample_indices[shap_probas.argmax()] # Highest churn probability
    low_risk_idx = evaluator.shap_sample_indices[shap_probas.argmin()] # Lowest churn probability

    print(f"   High-risk customer (index {high_risk_idx}): {evaluator.y_pred_proba[high_risk_idx]:.2%} churn prob")
    print(f"   Low-risk customer (index {low_risk_idx}): {evaluator.y_pred_proba[low_risk_idx]:.2%} churn prob")

    evaluator.plot_shap_waterfall(high_risk_idx)
    evaluator.plot_shap_waterfall(low_risk_idx)

    # Dependence plots for key features
    print("\nSHAP Dependence Plots:")

    # plot top 3 features
    for feature in top_features.head(3)['feature']:
        evaluator.plot_shap_dependence(feature)
    
    # ==========================================================
    # 9. Error Analysis
    # ==========================================================
    print("\n" + "=" * 70)
    print("STEP 6: Error Analysis")
    print("=" * 70)

    error_df = evaluator.analyze_errors()
    evaluator.plot_error_distribution()

    # ==========================================================
    # 10. Threshold Optimization
    # ==========================================================
    print("\n" + "=" * 70)
    print("STEP 7: Classification Threshold Optimization")
    print("=" * 70)

    # Business costs (same as experimentation report)
    optimal_threshold, optimal_metrics = evaluator.find_optimal_threshold(
        cost_fp=50,     # Cost of false positive (wasted campaign)
        cost_fn=600,    # Cost of false negative (0.6 * $1000 LTV)
    )

    # Save optimal threshold
    threshold_info = {
        'optimal_threshold': optimal_threshold,
        'default_threshold': 0.5,
        'cost_fp': 50,
        'cost_fn': 600,
        **optimal_metrics
    }

    import json
    with open('evaluation_results/optimal_threshold.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)
    
    print("Saved threshold info to evaluation_results/optimal_threshold.json")

    # ==========================================================
    # 11. Summary Plot
    # ==========================================================
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)

    print("\nAll results saved to: evaluation_results/")
    print("\nGenerated Artifacts:")
    print("   - Confusion matrices (2)")
    print("   - ROC curve")
    print("   - Precision-Recall curve")
    print("   - Calibration curve")
    print("   - SHAP summary plot")
    print("   - SHAP waterfall plots (2)")
    print("   - SHAP dependence plots (3)")
    print("   - Error distribution plot")
    print("   - Threshold optimization plot")
    print("   - Error analysis CSV")
    print("   - Optimal threshold JSON")

    print("\nNext Steps:")
    print("   1. Review SHAP plots for interpretability")
    print("   2. Analyze error patterns from 'error_analysis.csv'")
    print(f"   3. Consider using optimal threshold ({optimal_threshold:.3f}) instead of 0.5")
    print("   4. Document findings in evaluation report")
    print("   5. Proceed to Milestone 2.4 (Model Registry)")


if __name__ == "__main__":
    main()