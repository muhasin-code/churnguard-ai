"""
Feature engineering transformers for ChurnGuard AI.

Contains custom transformers for creating derived features.
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.base_transformer import BaseTransformer


class TenureBucketer(BaseTransformer):
    """
    Create tenure buckets (New, Growing, Established, Loyal, Veteran).
    
    Bins continuous tenure into categorical buckets based on
    customer lifecycle stages.
    """
    
    def __init__(self, bins: list, labels: list, encode_onehot: bool = True):
        """
        Initialize tenure bucketer.
        
        Args:
            bins: Bin edges (e.g., [0, 6, 12, 24, 60, 100])
            labels: Bin labels (e.g., ['New', 'Growing', ...])
            encode_onehot: If True, one-hot encode buckets
        """
        super().__init__()
        self.bins = bins
        self.labels = labels
        self.encode_onehot = encode_onehot
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit bucketer (no parameters to learn)."""
        self._validate_input(X)
        
        if 'Tenure' not in X.columns:
            raise ValueError("Column 'Tenure' not found in DataFrame")
        
        self.feature_names_in_ = list(X.columns)
        
        if self.encode_onehot:
            # Will create: TenureBucket_Growing, TenureBucket_Established, etc.
            # (drop first to avoid dummy trap)
            self.feature_names_out_ = (
                [col for col in X.columns if col != 'Tenure'] +
                [f'TenureBucket_{label}' for label in self.labels[1:]]
            )
        else:
            self.feature_names_out_ = list(X.columns) + ['TenureBucket']
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create tenure buckets."""
        self._check_is_fitted()
        self._validate_input(X)
        
        X = X.copy()
        
        # Create buckets
        X['TenureBucket'] = pd.cut(
            X['Tenure'],
            bins=self.bins,
            labels=self.labels,
            include_lowest=True
        )
        
        if self.encode_onehot:
            # One-hot encode buckets
            dummies = pd.get_dummies(
                X['TenureBucket'],
                prefix='TenureBucket',
                drop_first=True  # Avoid multicollinearity
            )
            # Convert bool columns to int for ML compatibility
            dummies = dummies.astype(int)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=['TenureBucket'])
        
        return X


class PriceToServiceRatio(BaseTransformer):
    """
    Calculate price-to-service ratio.
    
    Formula: monthly_charges / (num_services + 1)
    
    Identifies customers paying high prices for few services.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit (calculate number of services from data)."""
        self._validate_input(X)
        
        # Required columns
        required = ['MonthlyCharges', 'InternetService']
        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(X.columns) + ['PricePerService']
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-to-service ratio."""
        self._check_is_fitted()
        self._validate_input(X)
        
        X = X.copy()
        
        # Count number of services
        # (In real data, you'd count: internet, phone, streaming, etc.)
        # For our synthetic data, approximate from existing features
        num_services = (
            (X['InternetService'] != 'No Service').astype(int) +
            # Add other service indicators here if available
            1  # Base service (account)
        )
        
        X['PricePerService'] = X['MonthlyCharges'] / (num_services + 1)
        
        return X


class HighRiskSegment(BaseTransformer):
    """
    Flag high-risk customers based on combination of factors.
    
    High risk = low tenure + high charges + recent support issues
    """
    
    def __init__(self):
        """Initialize high risk segmenter."""
        super().__init__()
        self.monthly_charges_median_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit (learn median monthly charges from training data)."""
        self._validate_input(X)
        
        required = ['Tenure', 'MonthlyCharges', 'RecentSupportTickets']
        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Learn median from training data (avoid data leakage)
        self.monthly_charges_median_ = X['MonthlyCharges'].median()
        
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(X.columns) + ['IsHighRisk']
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Flag high-risk customers."""
        self._check_is_fitted()
        self._validate_input(X)
        
        X = X.copy()
        
        # High risk criteria
        is_high_risk = (
            (X['Tenure'] < 12) &
            (X['MonthlyCharges'] > self.monthly_charges_median_) &
            (X['RecentSupportTickets'] == 1)
        )
        
        X['IsHighRisk'] = is_high_risk.astype(int)
        
        return X


class ContractTenureMismatch(BaseTransformer):
    """
    Flag contract-tenure mismatches.
    
    Identifies long-term customers on month-to-month contracts
    (possible exit planning).
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit (no parameters to learn)."""
        self._validate_input(X)
        
        required = ['ContractType', 'Tenure']
        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(X.columns) + ['ContractTenureMismatch']
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Flag contract-tenure mismatches."""
        self._check_is_fitted()
        self._validate_input(X)
        
        X = X.copy()
        
        # Mismatch = month-to-month contract + long tenure
        is_mismatch = (
            (X['ContractType'] == 'Month-to-Month') &
            (X['Tenure'] > 24)
        )
        
        X['ContractTenureMismatch'] = is_mismatch.astype(int)
        
        return X


class FinancialStressScore(BaseTransformer):
    """
    Calculate financial stress score.
    
    Combines late payments and complaints into single distress metric.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit (no parameters to learn)."""
        self._validate_input(X)
        
        required = ['LatePayments', 'Complaints']
        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = list(X.columns) + ['FinancialStress']
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate financial stress score."""
        self._check_is_fitted()
        self._validate_input(X)
        
        X = X.copy()
        
        # Weighted combination
        X['FinancialStress'] = (
            X['LatePayments'] +
            (X['Complaints'] * 0.5)  # Complaints weighted less
        )
        
        return X