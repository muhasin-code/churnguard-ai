"""
Main feature engineering pipeline for ChurnGuard AI.

Orchestrates all feature transformations in the correct order.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.features.base_transformer import ColumnDropper, DataFrameWrapper, BaseTransformer
from src.features.feature_engineers import (
    TenureBucketer,
    PriceToServiceRatio,
    HighRiskSegment,
    ContractTenureMismatch,
    FinancialStressScore
)


class ChurnFeatureEngineer:
    """
    Main feature engineering pipeline for churn prediction.
    
    Handles:
    - Dropping unnecessary features
    - Creating engineered features
    - Encoding categorical variables
    - Scaling numerical features
    - Encoding target variable
    
    Example:
        engineer = ChurnFeatureEngineer()
        X_train_transformed, y_train = engineer.fit_transform(X_train, y_train)
        X_test_transformed, y_test = engineer.transform(X_test, y_test)
    """
    
    def __init__(self, config_path: str = "configs/feature_config.yaml"):
        """
        Initialize feature engineer.
        
        Args:
            config_path: Path to feature configuration YAML
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Will be set during fit()
        self.is_fitted_ = False
        self.feature_pipeline_ = None
        self.target_encoder_ = None
        self.feature_names_out_ = None
    
    def _load_config(self) -> dict:
        """Load feature configuration from YAML."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit feature transformations on training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self (for method chaining)
        """
        print("Fitting feature engineering pipeline...")
        
        # Build pipeline
        self.feature_pipeline_ = self._build_pipeline()
        
        # Fit feature transformations
        self.feature_pipeline_.fit(X, y)
        
        # Fit target encoder
        self.target_encoder_ = LabelEncoder()
        self.target_encoder_.fit(y)
        
        # Store output feature names
        self.feature_names_out_ = self._get_feature_names(X)
        
        self.is_fitted_ = True
        print(f"Pipeline fitted. Output features: {len(self.feature_names_out_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform
            y: Target to encode (optional)
            
        Returns:
            Tuple of (X_transformed, y_encoded)
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Transform features
        X_transformed = self.feature_pipeline_.transform(X)
        
        # Encode target if provided
        y_encoded = None
        if y is not None:
            y_encoded = self.target_encoder_.transform(y)
        
        return X_transformed, y_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Tuple of (X_transformed, y_encoded)
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def _build_pipeline(self) -> Pipeline:
        """
        Build the feature engineering pipeline.
        
        Returns:
            Sklearn Pipeline object
        """
        steps = []
        
        # Step 1: Drop unnecessary columns
        if self.config['features_to_drop']:
            steps.append((
                'drop_columns',
                ColumnDropper(columns=self.config['features_to_drop'])
            ))
        
        # Step 2: Create engineered features
        if self.config['engineered_features']['tenure_buckets']['enabled']:
            steps.append((
                'tenure_buckets',
                TenureBucketer(
                    bins=self.config['engineered_features']['tenure_buckets']['bins'],
                    labels=self.config['engineered_features']['tenure_buckets']['labels'],
                    encode_onehot=self.config['engineered_features']['tenure_buckets']['encode_as_onehot']
                )
            ))
        
        if self.config['engineered_features']['price_to_service_ratio']['enabled']:
            steps.append((
                'price_to_service',
                PriceToServiceRatio()
            ))
        
        if self.config['engineered_features']['high_risk_segment']['enabled']:
            steps.append((
                'high_risk',
                HighRiskSegment()
            ))
        
        if self.config['engineered_features']['contract_tenure_mismatch']['enabled']:
            steps.append((
                'contract_mismatch',
                ContractTenureMismatch()
            ))
        
        if self.config['engineered_features']['financial_stress_score']['enabled']:
            steps.append((
                'financial_stress',
                FinancialStressScore()
            ))
        
        # Step 3: Encode categorical variables
        # (This will be a custom transformer)
        steps.append((
            'encode_categoricals',
            CategoricalEncoder(config=self.config['categorical_features'])
        ))
        
        # Step 4: Scale numerical features
        steps.append((
            'scale_features',
            NumericalScaler(
                columns=self.config['numerical_features']['scale'],
                method=self.config['scaling']['method']
            )
        ))
        
        return Pipeline(steps)
    
    def _get_feature_names(self, X: pd.DataFrame) -> list:
        """
        Get output feature names after all transformations.
        
        Args:
            X: Original DataFrame
            
        Returns:
            List of output feature names
        """
        # Transform a sample to get column names
        X_sample = X.head(1)
        X_transformed = self.feature_pipeline_.transform(X_sample)
        
        return list(X_transformed.columns)
    
    def save(self, filepath: str):
        """
        Save fitted pipeline to disk.
        
        Args:
            filepath: Where to save (e.g., 'models/feature_pipeline.pkl')
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted pipeline")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'config': self.config,
            'feature_pipeline': self.feature_pipeline_,
            'target_encoder': self.target_encoder_,
            'feature_names_out': self.feature_names_out_
        }
        
        joblib.dump(save_dict, filepath)
        print(f"Saved feature pipeline to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load fitted pipeline from disk.
        
        Args:
            filepath: Path to saved pipeline
            
        Returns:
            Loaded ChurnFeatureEngineer instance
        """
        save_dict = joblib.load(filepath)
        
        # Create instance
        engineer = cls.__new__(cls)  # Skip __init__
        engineer.config = save_dict['config']
        engineer.feature_pipeline_ = save_dict['feature_pipeline']
        engineer.target_encoder_ = save_dict['target_encoder']
        engineer.feature_names_out_ = save_dict['feature_names_out']
        engineer.is_fitted_ = True
        
        print(f"Loaded feature pipeline from {filepath}")
        return engineer


# Helper transformers for the pipeline

class CategoricalEncoder(BaseTransformer):
    """
    Encode all categorical features based on config.
    
    Handles binary encoding and one-hot encoding.
    """
    
    def __init__(self, config: dict):
        """
        Initialize encoder.
        
        Args:
            config: Categorical features config from YAML
        """
        super().__init__()
        self.config = config
        self.encoders_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit encoders."""
        self._validate_input(X)
        
        self.feature_names_in_ = list(X.columns)
        
        # Fit binary encoders
        if 'binary' in self.config:
            for col, spec in self.config['binary'].items():
                if col in X.columns:
                    encoder = LabelEncoder()
                    encoder.fit(X[col])
                    self.encoders_[col] = encoder
        
        self.feature_names_out_ = self._calculate_output_features(X)
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        self._check_is_fitted()
        self._validate_input(X)
        
        X = X.copy()
        
        # Binary encoding
        if 'binary' in self.config:
            for col, spec in self.config['binary'].items():
                if col in X.columns and col in self.encoders_:
                    X[col] = self.encoders_[col].transform(X[col])
        
        # One-hot encoding
        if 'one_hot' in self.config:
            for col, spec in self.config['one_hot'].items():
                if col in X.columns:
                    dummies = pd.get_dummies(
                        X[col],
                        prefix=col,
                        drop_first=spec.get('drop_first', True)
                    )
                    # Convert bool columns to int for ML compatibility
                    dummies = dummies.astype(int)
                    X = pd.concat([X, dummies], axis=1)
                    X = X.drop(columns=[col])
        
        return X
    
    def _calculate_output_features(self, X: pd.DataFrame) -> list:
        """Calculate output feature names."""
        output_features = []
        
        for col in X.columns:
            # Check if this column gets encoded
            if 'binary' in self.config and col in self.config['binary']:
                output_features.append(col)  # Same name, just encoded
            elif 'one_hot' in self.config and col in self.config['one_hot']:
                # Will become multiple columns
                spec = self.config['one_hot'][col]
                categories = spec['categories']
                if spec.get('drop_first', True):
                    categories = categories[1:]  # Drop first category
                output_features.extend([f"{col}_{cat}" for cat in categories])
            else:
                output_features.append(col)  # Not encoded, keep as-is
        
        return output_features


class NumericalScaler(BaseTransformer):
    """
    Scale numerical features.
    
    Wraps sklearn scalers but preserves DataFrame structure.
    """
    
    def __init__(self, columns: list, method: str = 'standard'):
        """
        Initialize scaler.
        
        Args:
            columns: Columns to scale
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        super().__init__()
        self.columns = columns
        self.method = method
        
        # Create appropriate scaler
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit scaler on specified columns."""
        self._validate_input(X)
        
        # Only fit on columns that exist
        self.columns_to_scale_ = [col for col in self.columns if col in X.columns]
        
        if self.columns_to_scale_:
            self.scaler.fit(X[self.columns_to_scale_])
        
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(X.columns)
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale specified columns."""
        self._check_is_fitted()
        self._validate_input(X)
        
        X = X.copy()
        
        if self.columns_to_scale_:
            X[self.columns_to_scale_] = self.scaler.transform(X[self.columns_to_scale_])
        
        return X