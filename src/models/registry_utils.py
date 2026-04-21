"""
MLflow Model Registry utilities for ChurnGuard AI.

provides high-level API for registering, versioninig, and managing models in MLflow Model Registery with production deployment workflows.
"""


import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, List
import pandas as pd


class ModelRegistry:
    """
    Wrapper for MLflow Model Registry operations.

    Example:
        registry = ModelRegistry()
        registry.register_model(
            run_id="abc123",
            model_name="churnguard-xgboost,
            descriptions="XGBoost Conservative (v3.1 data)"
        )
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize registry client.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

        print(f"Connected to MLflow: {self.tracking_uri}")
    
    # =========================================================================
    # Model Registration
    # =========================================================================

    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Register a model from an MLflow run.

        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            description: Human-readable description
            tags: Additional metadata tags
        
        Returns:
            Version number of regsitered model
        """
        print(f"\nRegistering model: {model_name}")
        print(f"   Source run: {run_id}")

        # Model URI pointing to the logged model artifact
        model_uri = f"runs:/{run_id}/model"

        # Register the model (creates new version)
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        version_number = model_version.version

        print(f"Registered as version {version_number}")

        # Update description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=version_number,
                description=description
            )
            print(f"   Added description")
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=version_number,
                    key=key,
                    value=value
                )
            print(f"   Added {len(tags)} tags")
        
        return version_number
    
    def register_champion(
        self,
        run_id: str,
        model_name: str = "churnguard-classifier",
        data_version: str = "v3.1",
        performance_metrics: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Register champion model with comprehensive metadata.

        Args:
            run_id: MLflow run ID
            model_name: Registered model name
            data_version: Version of training data (e.g, "v3.1")
            performance_metrics: Dict of metric_name: value
        
        Returns:
            Version number
        """
        # Get run details
        run = self.client.get_run(run_id=run_id)
        run_name = run.data.tags.get("mlflow.runName", "unknown")

        # Build descriptions
        description = f"""
Champion model from run: {run_name}

Training Details:
- Data Version: {data_version}
- Training Date: {run.info.start_time}
- Model Type: {run.data.params.get('model_type', 'XGBoost')}
        """.strip()

        if performance_metrics:
            description += "\n\nPerformance Metrics:\n"
            for metric, value in performance_metrics.items():
                description += f"- {metric}: {value}\n"
        
        # Build tags
        tags = {
            "data_version": data_version,
            "run_name": run_name,
            "model_type": run.data.params.get("model_type", "XGBoost"),
            "champion": "true"
        }

        # Add metrics as tags
        if performance_metrics:
            for metric, value in performance_metrics.items():
                tags[f"metric_{metric}"] = str(value)
        
        # Register
        version = self.register_model(
            run_id=run_id,
            model_name=model_name,
            description=description,
            tags=tags
        )

        return version
    
    # =========================================================================
    # Stage Management
    # =========================================================================

    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition a model version to a new stage.

        Args:
            model_name: Registered model name
            version: Version number to transition
            stage: Target stage ("Staging", "Production", "Archived", "None")
            archive_existing: If True, archive current production model
        """
        valid_stages = ["Staging", "Production", "Archived", "None"]
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")
        
        print(f"\nTransitioning {model_name} v{version} -> {stage}")

        # Archive existing production version if moving to production
        if stage == "Production" and archive_existing:
            existing_prod = self.get_production_version(model_name)
            if existing_prod:
                print(f"   Archiving current production: v{existing_prod}")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=existing_prod,
                    stage="Archived"
                )
        
        # Transition to new stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )

        print(f"Transitioned v{version} to {stage}")
    
    def promote_to_staging(self, model_name: str, version: int):
        """Promote model version to Staging."""
        self.transition_stage(model_name, version, "Staging", archive_existing=False)
    
    def promote_to_production(self, model_name: str, version: int):
        """Promote model version to Production."""
        self.transition_stage(model_name, version, "Production", archive_existing=True)
    
    def archive_version(self, model_name: str, version: int):
        """Archive a model version."""
        self.transition_stage(model_name, version, "Archived", archive_existing=False)
    
    # =========================================================================
    # Model Discovery
    # =========================================================================
    
    def list_registered_models(self) -> pd.DataFrame:
        """
        List all registered models.

        Returns:
            DataFrame wit model name and metadata
        """
        models = self.client.search_registered_models()

        if not models:
            print("No registered model found")
            return pd.DataFrame()
        
        data = []
        for model in models:
            latest_versions = model.latest_versions

            for version in latest_versions:
                data.append({
                    'name': model.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_time': version.creation_timestamp
                })
        
        df = pd.DataFrame(data)
        return df.sort_values(['name', 'version'], ascending=[True, False])
    
    def get_model_versions(self, model_name: str) -> pd.DataFrame:
        """
        Get all versions of a specific model.

        Args:
            model_name: Registered model name
        
        Returns:
            DataFrame with version details
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")

        if not versions:
            print(f"No versions found for {model_name}")
            return pd.DataFrame
        
        data = []
        for v in versions:
            # Get run to fetch metrics
            run = self.client.get_run(v.run_id)

            data.append({
                'version': v.version,
                'stage': v.current_stage,
                'run_id': v.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
                'roc_auc': run.data.metrics.get('test_roc_auc', None),
                'accuracy': run.data.metrics.get('test_accuracy', None),
                'created': v.creation_timestamp,
                'description': v.description[:50] + '...' if v.description and len(v.description) > 50 else v.description
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('version', ascending=False)
    
    def get_production_version(self, model_name: str) -> Optional[int]:
        """
        Get the current production version number.

        Args:
            model_name: Registered model name
        
        Returns:
            Version number or None if no production version
        """
        versions = self.client.get_latest_versions(model_name, stages=["Production"])

        if not versions:
            return None
        
        return versions[0].version
    
    def get_staging_version(self, model_name: str) -> Optional[int]:
        """Get the current staging version number."""
        versions = self.client.get_latest_versions(model_name, stages=["Staging"])
        return versions[0].version if versions else None
    
    # =========================================================================
    # Model Loading
    # =========================================================================
    
    def load_model(self, model_name: str, version: Optional[int] = None, stage: Optional[str] = None):
        """
        Load a registered model for inference.

        Args:
            model_name: Registered model name
            verions: Specific version number (optional)
            stage: Stage to load (optional)
        
        Returns: 
            Loaded model object
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
            print(f"Loading {model_name} version {version}")
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
            print(f"Loading {model_name} from {stage} stage")
        else:
            raise ValueError("Must specify either version or stage")
        
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully")

        return model
    
    # =========================================================================
    # Comparison & Analysis
    # =========================================================================
    
    def compare_versions(
        self,
        model_name: str,
        versions: List[int],
        metrics: List[str] = ["test_roc_auc", "test_accuracy", "test_precision", "test_recall"]
    ) -> pd.DataFrame:
        """
        Compare performance metrics accross multiple versions.

        Args:
            model_name: Registered model name
            versions: List of version numbers to compare
            metrics: List of metric names to compare
        
        Returns:
            DataFrame with comparison
        """
        data = []

        for version in versions:
            version_obj = self.client.get_model_version(model_name, version)
            run = self.client.get_run(version_obj.run_id)

            row = {
                'version': version,
                'stage': version_obj.current_stage,
                'run_name': run.data.tags.get('mlflow.runName', 'unknown')
            }

            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values('version')
    
    # =========================================================================
    # Metadata Management
    # =========================================================================
    
    def update_descriptions(self, model_name: str, version: int, description: str):
        """Update model version description."""
        self.client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        print(f"Updated description for {model_name} v{version}")
    
    def add_tags(self, model_name: str, version: int, tags: Dict[str, str]):
        """Add tags to model versions."""
        for key, value in tags.items():
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=key,
                value=value
            )
        print(f"Added {len(tags)} tags to {model_name} v{version}")
    
    def get_model_metadata(self, model_name: str, version: int) -> Dict:
        """
        Get comprehensive metadat for a model version.

        Returns:
            Dictionary with all metadata
        """
        version_obj = self.client.get_model_version(model_name, version)
        run = self.client.get_run(version_obj.run_id)

        metadata = {
            'model_name': model_name,
            'version': version,
            'stage': version_obj.current_stage,
            'run_id': version_obj.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
            'description': version_obj.description,
            'tags': version_obj.tags,
            'metrics': run.data.metrics,
            'parameters': run.data.params,
            'created_timestamp': version_obj.creation_timestamp,
            'last_updated_timestampe': version_obj.last_updated_timestamp
        }

        return metadata