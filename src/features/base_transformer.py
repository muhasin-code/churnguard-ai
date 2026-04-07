"""
Base transformer class for feature engineering.

Provides fit/transform interface compatible with scikit-learn pipelines.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Optional


class BaseTransformer(ABC):
    """
    Abstract base class for feature transformers.
    
    All transformers must implement fit() and transform() methods.
    This ensures compatibility with scikit-learn pipelines.
    """
    
    def __init__(self):
        """Initialize transformer."""
        self.is_fitted_ = False
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the transformer on training data.
        
        This method learns parameters from the training data
        (e.g., means for scaling, categories for encoding).
        
        Args:
            X: Training features
            y: Training target (optional, needed for target encoding)
            
        Returns:
            self (for method chaining)
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        This method applies transformations learned during fit().
        MUST NOT learn new parameters from X.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed DataFrame
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Convenience method for training data.
        Equivalent to: self.fit(X, y).transform(X)
        
        Args:
            X: Training features
            y: Training target (optional)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _check_is_fitted(self):
        """Verify transformer has been fitted."""
        if not self.is_fitted_:
            raise ValueError(
                f"{self.__class__.__name__} must be fitted before transform. "
                f"Call fit() or fit_transform() first."
            )
    
    def _validate_input(self, X: pd.DataFrame):
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
    
    def get_feature_names_out(self) -> list:
        """
        Get output feature names after transformation.
        
        Returns:
            List of feature names
        """
        self._check_is_fitted()
        return self.feature_names_out_


class ColumnDropper(BaseTransformer):
    """
    Drop specified columns from DataFrame.
    
    Example:
        dropper = ColumnDropper(columns=['CustomerID', 'ChurnProbability'])
        X_transformed = dropper.fit_transform(X)
    """
    
    def __init__(self, columns: list):
        """
        Initialize column dropper.
        
        Args:
            columns: List of column names to drop
        """
        super().__init__()
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the dropper (just validates columns exist)."""
        self._validate_input(X)
        
        # Check that columns to drop exist
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = [col for col in X.columns if col not in self.columns]
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns."""
        self._check_is_fitted()
        self._validate_input(X)
        
        return X.drop(columns=self.columns, errors='ignore')


class DataFrameWrapper:
    """
    Utility to convert sklearn transformers to DataFrame-friendly versions.
    
    Wraps sklearn transformers (like StandardScaler) to preserve column names.
    """
    
    def __init__(self, transformer, columns: list):
        """
        Initialize wrapper.
        
        Args:
            transformer: sklearn transformer (e.g., StandardScaler())
            columns: Columns to apply transformer to
        """
        self.transformer = transformer
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit on specified columns."""
        self.transformer.fit(X[self.columns], y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform specified columns, preserve others."""
        X = X.copy()
        X[self.columns] = self.transformer.transform(X[self.columns])
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)