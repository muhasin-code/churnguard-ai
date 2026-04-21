"""
Register champion model in MLflow Model Registry.

This script takes the best-performing model from experimentation and registers it in the MLflow Model Registry with comprehensive metadata for production deployment.
"""


import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry_utils import ModelRegistry


def main():
    """Register champion model with metadata."""
    
    print("=" * 70)
    print("ChurnGuard AI - Champion Model Registration")
    print("=" * 70)

    # Initialize registry
    registry = ModelRegistry(tracking_uri="http://localhost:5000")

    # =========================================================================
    # CHAMPION MODEL DETAILS
    # =========================================================================
    
    # TODO: Update this with your actual champion run ID
    champion_run_id = "20e971f66ba64291a8a6f6763418f5ed" # cgb-conservative

    model_name = "churnguard-classifier"
    data_version = "v3.1"

    # Performance metrics from evaluation
    performance_metrics = {
        "roc_auc": 0.8077,
        "accuracy": 0.7366,
        "precision": 0.8360,
        "recall": 0.7414,
        "f1_score": 0.7859,
        "pr_auc": 0.8786
    }

    # =========================================================================
    # REGISTER MODEL
    # =========================================================================
    
    print("\nRegistering champion model...")
    print(f"   Run ID: {champion_run_id}")
    print(f"   Model Name: {model_name}")
    print(f"   Data Version: {data_version}")

    version = registry.register_champion(
        run_id=champion_run_id,
        model_name=model_name,
        data_version=data_version,
        performance_metrics=performance_metrics
    )

    print(f"\nChampion model registered as version {version}")

    # =========================================================================
    # PROMOTE TO STAGING
    # =========================================================================
    
    print("\nPromoting to Staging...")
    registry.promote_to_staging(model_name, version)

    print(f"\nModel v{version} is now in Staging")
    print("   Ready for validation testing")

    # =========================================================================
    # DISPLAY MODEL INFO
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("MODEL REGISTER STRATEGY")
    print("=" * 70)

    # List all versions
    version_df = registry.get_model_versions(model_name)
    print(f"\nAll versions of {model_name}:")
    print(version_df.to_string(index=False))

    # Get metadata
    metadata = registry.get_model_metadata(model_name, version)

    print("\n" + "=" * 70)
    print(f"CHAMPION MODEL METADATA (v{version})")
    print("=" * 70)

    print(f"\nStage: {metadata['stage']}")
    print(f"Run Name: {metadata['run_name']}")
    print(f"Run ID: {metadata['run_id']}")

    print("\nPerformance Metrics")
    for metric, value in sorted(metadata['metrics'].items()):
        if metric.startswith('test_'):
            print(f"   {metric}: {value:.4f}")
    
    print("\nTags:")
    for key, value in sorted(metadata['tags'].items()):
        print(f"   {key}: {value}")
    
    # =========================================================================
    # NEXT STEPS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    print(f"""
1. Model registered in MLflow Registry
2. Currently in 'Staging' stage
3. Validate model performance in staging environment
4. If validation passes, promote to Production:
   
   from src.models.registry_utils import ModelRegistry
   registry = ModelRegistry()
   registry.promote_to_production('{model_name}', {version})

5. Deploy to production API (Milestone 3.1)

View in MLflow UI: http://localhost:5000/#/models/{model_name}
    """)


if __name__ == "__main__":
    main()