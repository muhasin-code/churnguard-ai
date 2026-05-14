"""
MLflow model loading service.

Handles loading the production model from MLflow Registry with caching.
"""

import mlflow
import mlflow.sklearn
import joblib
from typing import Optional, Any
from pathlib import Path
import time

from src.api.config import settings, get_project_root
from src.api.utils.logging import api_logger


class ModelLoader:
    """
    Loads and caches ML models from MLflow Registry.
    
    Implements singleton pattern to ensure only one model instance
    is loaded in memory.
    
    Example:
        loader = ModelLoader()
        model = loader.get_model()
        predictions = model.predict(X)
    """
    
    _instance = None
    _model = None
    _feature_pipeline = None
    _model_metadata = {}
    _last_load_time = None

    _target_encoder = None
    _feature_names = None
    _pipeline_config = None
    
    def __new__(cls):
        """Singleton pattern - only one instance."""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model loader."""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            api_logger.info("ModelLoader initialized")
    
    # =========================================================================
    # Model Loading
    # =========================================================================
    
    def load_model_from_registry(self) -> Any:
        """
        Load model from MLflow Model Registry.
        
        Returns:
            Loaded sklearn model
            
        Raises:
            Exception: If model loading fails
        """
        api_logger.info("=" * 70)
        api_logger.info("Loading model from MLflow Registry")
        api_logger.info("=" * 70)
        
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            api_logger.info(f"MLflow URI: {settings.mlflow_tracking_uri}")
            
            # Construct model URI
            model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
            api_logger.info(f"Model URI: {model_uri}")
            
            # Load model
            api_logger.info("Loading model...")
            start_time = time.time()
            
            model = mlflow.sklearn.load_model(model_uri)
            
            load_time = time.time() - start_time
            api_logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            # Get model metadata
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            try:
                model_version = client.get_latest_versions(
                    settings.model_name,
                    stages=[settings.model_stage]
                )[0]
                
                self._model_metadata = {
                    "name": settings.model_name,
                    "version": model_version.version,
                    "stage": settings.model_stage,
                    "run_id": model_version.run_id,
                    "description": model_version.description
                }
                
                api_logger.info(f"Model metadata: {self._model_metadata}")
                
            except Exception as e:
                api_logger.warning(f"Could not fetch model metadata: {e}")
                self._model_metadata = {
                    "name": settings.model_name,
                    "version": "unknown",
                    "stage": settings.model_stage
                }
            
            self._last_load_time = time.time()
            
            return model
            
        except Exception as e:
            api_logger.error(f"Failed to load model from MLflow: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def load_feature_pipeline(self) -> Any:
        """
        Load feature engineering pipeline.
        
        Returns:
            Feature pipeline object
            
        Raises:
            FileNotFoundError: If pipeline file not found
        """
        api_logger.info("Loading feature pipeline...")
        
        try:
            pipeline_path = get_project_root() / settings.feature_pipeline_path
            
            if not pipeline_path.exists():
                raise FileNotFoundError(
                    f"Feature pipeline not found: {pipeline_path}\n"
                    f"Run: python scripts/engineer_features.py"
                )
            
            # Load the pipeline file
            loaded_obj = joblib.load(pipeline_path)
            
            # The file is a dict with structure:
            # {
            #   'feature_pipeline': sklearn.pipeline.Pipeline,
            #   'target_encoder': LabelEncoder,
            #   'feature_names_out': list,
            #   'config': dict
            # }
            
            if isinstance(loaded_obj, dict):
                # Extract the feature pipeline from the dict
                pipeline = loaded_obj['feature_pipeline']
                api_logger.info("Extracted 'feature_pipeline' from saved dict")
                
                # Store additional components for potential future use
                self._target_encoder = loaded_obj.get('target_encoder')
                self._feature_names = loaded_obj.get('feature_names_out')
                self._pipeline_config = loaded_obj.get('config')
                
                api_logger.info(f"   Feature names: {len(self._feature_names)} features")
                api_logger.info(f"   Has target encoder: {self._target_encoder is not None}")
                
            else:
                # Direct pipeline object (shouldn't happen with your setup, but handle it)
                pipeline = loaded_obj
                api_logger.info("Loaded pipeline object directly")
            
            # Verify pipeline has transform method
            if not hasattr(pipeline, 'transform'):
                raise ValueError(
                    f"Pipeline object does not have 'transform' method. "
                    f"Type: {type(pipeline)}"
                )
            
            api_logger.info(f"Feature pipeline loaded from {pipeline_path}")
            api_logger.info(f"   Pipeline type: {type(pipeline)}")
            
            return pipeline
            
        except Exception as e:
            api_logger.error(f"Failed to load feature pipeline: {str(e)}")
            raise RuntimeError(f"Feature pipeline loading failed: {str(e)}")
        
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_model(self, force_reload: bool = False) -> Any:
        """
        Get model (with caching).
        
        Args:
            force_reload: If True, reload model even if cached
            
        Returns:
            Loaded model
        """
        if self._model is None or force_reload:
            api_logger.info("Loading model (not cached or force reload)")
            self._model = self.load_model_from_registry()
        else:
            api_logger.debug("Using cached model")
        
        return self._model
    
    def get_feature_pipeline(self, force_reload: bool = False) -> Any:
        """
        Get feature pipeline (with caching).
        
        Args:
            force_reload: If True, reload pipeline even if cached
            
        Returns:
            Feature pipeline
        """
        if self._feature_pipeline is None or force_reload:
            api_logger.info("Loading feature pipeline (not cached or force reload)")
            self._feature_pipeline = self.load_feature_pipeline()
        else:
            api_logger.debug("Using cached feature pipeline")
        
        return self._feature_pipeline
    
    def get_model_metadata(self) -> dict:
        """
        Get model metadata.
        
        Returns:
            Dict with model name, version, stage, etc.
        """
        return self._model_metadata.copy()
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def is_feature_pipeline_loaded(self) -> bool:
        """Check if feature pipeline is loaded."""
        return self._feature_pipeline is not None
    
    def get_load_time(self) -> Optional[float]:
        """Get timestamp of last model load."""
        return self._last_load_time
    
    def reload_model(self):
        """Force reload model from registry."""
        api_logger.info("Forcing model reload...")
        self._model = None
        self._feature_pipeline = None
        self._model_metadata = {}
        return self.get_model(force_reload=True)


# ============================================================================
# Global Instance
# ============================================================================

# Create global model loader instance
model_loader = ModelLoader()