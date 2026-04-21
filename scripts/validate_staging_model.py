"""
Validate staging model before production promotion.

Performs comprehensive validation to ensure model is ready
for production deployment. This is the "quality gate" before
promoting from Staging → Production.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry_utils import ModelRegistry


def validate_model_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_roc_auc: float = 0.80,
    min_recall: float = 0.70
) -> bool:
    """
    Validate model meets minimum performance criteria.
    
    Args:
        model: Loaded model
        X_test: Test features
        y_test: Test labels
        min_roc_auc: Minimum acceptable ROC-AUC
        min_recall: Minimum acceptable recall
        
    Returns:
        True if all criteria met
    """
    from sklearn.metrics import roc_auc_score, recall_score, accuracy_score
    
    print("\nPERFORMANCE VALIDATION")
    print("=" * 70)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Check criteria
    criteria = {
        'ROC-AUC': {
            'value': roc_auc,
            'threshold': min_roc_auc,
            'pass': roc_auc >= min_roc_auc
        },
        'Recall': {
            'value': recall,
            'threshold': min_recall,
            'pass': recall >= min_recall
        }
    }
    
    # Display results
    all_pass = True
    for metric_name, metric_info in criteria.items():
        status = "PASS" if metric_info['pass'] else "FAIL"
        print(f"{metric_name:15} {metric_info['value']:.4f} (min: {metric_info['threshold']:.2f})  {status}")
        
        if not metric_info['pass']:
            all_pass = False
    
    print(f"\nAccuracy:       {accuracy:.4f} (informational)")
    
    return all_pass


def validate_model_signature(model) -> bool:
    """
    Validate model has proper input/output signature.
    
    Args:
        model: Loaded model
        
    Returns:
        True if signature is valid
    """
    print("\nSIGNATURE VALIDATION")
    print("=" * 70)
    
    # Check if model has required methods
    required_methods = ['predict', 'predict_proba']
    
    for method in required_methods:
        if not hasattr(model, method):
            print(f"FAIL: Missing method '{method}'")
            return False
        print(f"Method '{method}' exists")
    
    return True


def validate_prediction_sanity(
    model,
    X_test: pd.DataFrame
) -> bool:
    """
    Sanity check predictions (no NaN, valid probability range).
    
    Args:
        model: Loaded model
        X_test: Test features
        
    Returns:
        True if predictions are sane
    """
    print("\nPREDICTION SANITY CHECKS")
    print("=" * 70)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Check 1: No NaN predictions
    if np.any(np.isnan(y_pred_proba)):
        print("FAIL: NaN values in predictions")
        return False
    print("No NaN values in predictions")
    
    # Check 2: Valid probability range
    if np.any((y_pred_proba < 0) | (y_pred_proba > 1)):
        print("FAIL: Probabilities outside [0, 1] range")
        return False
    print("Probabilities in valid range [0, 1]")
    
    # Check 3: Reasonable distribution
    mean_prob = y_pred_proba.mean()
    if mean_prob < 0.1 or mean_prob > 0.9:
        print(f"WARNING: Mean probability {mean_prob:.2f} seems extreme")
    else:
        print(f"Mean probability: {mean_prob:.2f} (reasonable)")
    
    # Check 4: Variance in predictions
    if y_pred_proba.std() < 0.05:
        print("WARNING: Very low variance in predictions")
    else:
        print(f"Prediction variance: {y_pred_proba.std():.3f} (good)")
    
    return True


def main():
    """Validate staging model for production promotion."""
    
    print("=" * 70)
    print("ChurnGuard AI - Staging Model Validation")
    print("=" * 70)
    
    # Initialize registry
    registry = ModelRegistry(tracking_uri="http://localhost:5000")
    
    model_name = "churnguard-classifier"
    
    # =========================================================================
    # 1. Load staging model
    # =========================================================================
    
    print("\nLoading staging model...")
    
    staging_version = registry.get_staging_version(model_name)
    
    if not staging_version:
        print(f"No model in Staging stage for '{model_name}'")
        print("   Register a model first with scripts/register_champion.py")
        sys.exit(1)
    
    print(f"   Found version {staging_version} in Staging")
    
    model = registry.load_model(model_name, stage="Staging")
    
    # =========================================================================
    # 2. Load test data
    # =========================================================================
    
    print("\nLoading test data...")
    
    test_df = pd.read_csv("data/processed/features_v1_test.csv")
    
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    print(f"   Test set: {X_test.shape}")
    
    # =========================================================================
    # 3. Run validation checks
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("RUNNING VALIDATION CHECKS")
    print("=" * 70)
    
    checks = {
        'signature': validate_model_signature(model),
        'sanity': validate_prediction_sanity(model, X_test),
        'performance': validate_model_performance(
            model, X_test, y_test,
            min_roc_auc=0.80,
            min_recall=0.70
        )
    }
    
    # =========================================================================
    # 4. Validation summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check_name.upper():20} {status}")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("ALL VALIDATION CHECKS PASSED")
        print(f"\n   Model v{staging_version} is ready for production!")
        print("\n   To promote to production:")
        print(f"   >>> from src.models.registry_utils import ModelRegistry")
        print(f"   >>> registry = ModelRegistry()")
        print(f"   >>> registry.promote_to_production('{model_name}', {staging_version})")
        print("\n   Or run: python scripts/promote_to_production.py")
    else:
        print("VALIDATION FAILED")
        print(f"\n   Model v{staging_version} NOT ready for production")
        print("   Please fix issues and re-validate")
        sys.exit(1)


if __name__ == "__main__":
    main()