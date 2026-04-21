"""
Promote validated model from Staging to Production.

This script moves a model from Staging → Production stage
after validation has passed. It automatically archives the
previous production version.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry_utils import ModelRegistry


def main():
    """Promote staging model to production."""
    
    print("=" * 70)
    print("ChurnGuard AI - Production Promotion")
    print("=" * 70)
    
    # Initialize registry
    registry = ModelRegistry(tracking_uri="http://localhost:5000")
    
    model_name = "churnguard-classifier"
    
    # Get staging version
    staging_version = registry.get_staging_version(model_name)
    
    if not staging_version:
        print(f"No model in Staging for '{model_name}'")
        print("   Validate model first with scripts/validate_staging_model.py")
        sys.exit(1)
    
    print(f"\nStaging model: v{staging_version}")
    
    # Confirm promotion
    print("\nThis will:")
    print(f"   1. Promote v{staging_version} to Production")
    print(f"   2. Archive current production version (if any)")
    
    confirm = input("\nProceed? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Promotion cancelled")
        sys.exit(0)
    
    # Promote
    print(f"\nPromoting v{staging_version} to Production...")
    
    registry.promote_to_production(model_name, staging_version)
    
    print("\nModel promoted to Production!")
    
    # Display current state
    print("\n" + "=" * 70)
    print("CURRENT REGISTRY STATE")
    print("=" * 70)
    
    versions_df = registry.get_model_versions(model_name)
    print(versions_df[['version', 'stage', 'run_name', 'roc_auc']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("PRODUCTION MODEL DEPLOYED")
    print("=" * 70)
    
    print(f"""
Model v{staging_version} is now in Production!

Next steps:
1. Deploy to production API (Milestone 3.1)
2. Set up monitoring (Milestone 4.1)
3. Monitor performance and data drift

View in MLflow UI: http://localhost:5000/#/models/{model_name}
    """)


if __name__ == "__main__":
    main()