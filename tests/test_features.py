"""
Unit tests for feature engineering module.

Tests all custom transformers and the main pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import ChurnFeatureEngineer
from src.features.feature_engineers import (
    TenureBucketer,
    PriceToServiceRatio,
    HighRiskSegment,
    ContractTenureMismatch,
    FinancialStressScore
)
from src.features.base_transformer import ColumnDropper


# ============================================
# Fixtures (Test Data)
# ============================================

@pytest.fixture
def sample_data():
    """Create small sample dataset for testing."""
    return pd.DataFrame({
        'CustomerID': ['CUST001', 'CUST002', 'CUST003', 'CUST004'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Age': [25, 45, 35, 60],
        'Tenure': [3, 18, 30, 65],
        'ContractType': ['Month-to-Month', 'One Year', 'Two Year', 'Month-to-Month'],
        'InternetService': ['Fiber', 'DSL', 'No Service', 'Fiber'],
        'MonthlyCharges': [85.0, 65.0, 45.0, 95.0],
        'TotalCharges': [255.0, 1170.0, 1350.0, 6175.0],
        'CallMinutes': [450.0, 380.0, 200.0, 520.0],
        'DataUsage': [5.2, 3.8, 0.0, 6.5],
        'Complaints': [1, 0, 0, 2],
        'RecentSupportTickets': [1, 0, 0, 0],
        'PaymentMethod': ['Credit Card', 'UPI', 'Cash', 'Debit Card'],
        'LatePayments': [0, 0, 1, 2],
        'Engagement': [0.95, 0.88, 0.20, 1.17],
        'ChurnProbability': [0.75, 0.35, 0.25, 0.65],
        'Churn': ['Yes', 'No', 'No', 'Yes']
    })


@pytest.fixture
def sample_X_y(sample_data):
    """Split sample data into X and y."""
    X = sample_data.drop(columns=['Churn'])
    y = sample_data['Churn']
    return X, y


# ============================================
# Test ColumnDropper
# ============================================

def test_column_dropper_drops_specified_columns(sample_data):
    """Test that ColumnDropper removes specified columns."""
    dropper = ColumnDropper(columns=['CustomerID', 'ChurnProbability'])
    
    result = dropper.fit_transform(sample_data)
    
    assert 'CustomerID' not in result.columns
    assert 'ChurnProbability' not in result.columns
    assert len(result.columns) == len(sample_data.columns) - 2


def test_column_dropper_preserves_other_columns(sample_data):
    """Test that ColumnDropper keeps non-specified columns."""
    dropper = ColumnDropper(columns=['CustomerID'])
    
    result = dropper.fit_transform(sample_data)
    
    assert 'Gender' in result.columns
    assert 'Age' in result.columns
    assert len(result) == len(sample_data)  # Same number of rows


def test_column_dropper_handles_missing_columns(sample_data):
    """Test that ColumnDropper handles columns that don't exist gracefully."""
    dropper = ColumnDropper(columns=['CustomerID', 'NonExistentColumn'])
    
    # Should not raise error during fit
    dropper.fit(sample_data)
    
    # Should ignore non-existent column during transform
    result = dropper.transform(sample_data)
    assert 'CustomerID' not in result.columns


# ============================================
# Test TenureBucketer
# ============================================

def test_tenure_bucketer_creates_buckets():
    """Test that TenureBucketer creates tenure categories."""
    df = pd.DataFrame({
        'Tenure': [2, 10, 20, 40, 70]
    })
    
    bucketer = TenureBucketer(
        bins=[0, 6, 12, 24, 60, 100],
        labels=['New', 'Growing', 'Established', 'Loyal', 'Veteran'],
        encode_onehot=False
    )
    
    result = bucketer.fit_transform(df)
    
    assert 'TenureBucket' in result.columns
    assert result['TenureBucket'].iloc[0] == 'New'  # Tenure=2
    assert result['TenureBucket'].iloc[2] == 'Established'  # Tenure=20


def test_tenure_bucketer_onehot_encoding():
    """Test that TenureBucketer one-hot encodes when requested."""
    df = pd.DataFrame({
        'Tenure': [2, 10, 20, 40, 70]
    })
    
    bucketer = TenureBucketer(
        bins=[0, 6, 12, 24, 60, 100],
        labels=['New', 'Growing', 'Established', 'Loyal', 'Veteran'],
        encode_onehot=True
    )
    
    result = bucketer.fit_transform(df)
    
    # Should create one-hot columns (minus first category)
    assert 'TenureBucket_Growing' in result.columns
    assert 'TenureBucket_Established' in result.columns
    assert 'TenureBucket' not in result.columns  # Original dropped


# ============================================
# Test PriceToServiceRatio
# ============================================

def test_price_to_service_ratio_creates_feature(sample_data):
    """Test that PriceToServiceRatio creates the ratio feature."""
    transformer = PriceToServiceRatio()
    
    result = transformer.fit_transform(sample_data)
    
    assert 'PricePerService' in result.columns
    assert result['PricePerService'].notna().all()
    assert (result['PricePerService'] > 0).all()  # Should be positive


def test_price_to_service_ratio_calculation(sample_data):
    """Test that PriceToServiceRatio calculates correctly."""
    transformer = PriceToServiceRatio()
    
    result = transformer.fit_transform(sample_data)
    
    # For customer with Fiber internet (1 service) + base (1)
    # num_services = 2, so ratio should be charges / 3
    row_0 = result.iloc[0]
    expected_ratio = 85.0 / 3  # MonthlyCharges / (num_services + 1)
    
    assert abs(row_0['PricePerService'] - expected_ratio) < 0.01


# ============================================
# Test HighRiskSegment
# ============================================

def test_high_risk_segment_flags_risky_customers(sample_data):
    """Test that HighRiskSegment correctly identifies high-risk customers."""
    transformer = HighRiskSegment()
    
    result = transformer.fit_transform(sample_data)
    
    assert 'IsHighRisk' in result.columns
    
    # Customer 0: Tenure=3 (low), Charges=85 (high), RecentTicket=1
    # Should be flagged as high risk
    assert result['IsHighRisk'].iloc[0] == 1
    
    # Customer 2: Tenure=30 (high), should NOT be high risk
    assert result['IsHighRisk'].iloc[2] == 0


def test_high_risk_segment_learns_median_from_train():
    """Test that HighRiskSegment learns median from training data."""
    train = pd.DataFrame({
        'Tenure': [10, 20, 30],
        'MonthlyCharges': [50, 60, 70],
        'RecentSupportTickets': [0, 1, 0]
    })
    
    test = pd.DataFrame({
        'Tenure': [5],
        'MonthlyCharges': [65],  # Above train median (60)
        'RecentSupportTickets': [1]
    })
    
    transformer = HighRiskSegment()
    transformer.fit(train)  # Learns median=60 from train
    
    result = transformer.transform(test)
    
    # Should use train's median, not test's value
    assert transformer.monthly_charges_median_ == 60
    assert result['IsHighRisk'].iloc[0] == 1  # Tenure low + charges > train_median + ticket


# ============================================
# Test ContractTenureMismatch
# ============================================

def test_contract_tenure_mismatch_flags_correctly(sample_data):
    """Test that ContractTenureMismatch identifies mismatches."""
    transformer = ContractTenureMismatch()
    
    result = transformer.fit_transform(sample_data)
    
    assert 'ContractTenureMismatch' in result.columns
    
    # Customer 3: Month-to-Month + Tenure=65 → Mismatch
    assert result['ContractTenureMismatch'].iloc[3] == 1
    
    # Customer 0: Month-to-Month but Tenure=3 → No mismatch
    assert result['ContractTenureMismatch'].iloc[0] == 0
    
    # Customer 2: Two year contract → No mismatch
    assert result['ContractTenureMismatch'].iloc[2] == 0


# ============================================
# Test FinancialStressScore
# ============================================

def test_financial_stress_score_calculates_correctly(sample_data):
    """Test that FinancialStressScore calculates the weighted sum."""
    transformer = FinancialStressScore()
    
    result = transformer.fit_transform(sample_data)
    
    assert 'FinancialStress' in result.columns
    
    # Customer 0: LatePayments=0, Complaints=1
    # Score = 0 + (1 * 0.5) = 0.5
    assert result['FinancialStress'].iloc[0] == 0.5
    
    # Customer 3: LatePayments=2, Complaints=2
    # Score = 2 + (2 * 0.5) = 3.0
    assert result['FinancialStress'].iloc[3] == 3.0


# ============================================
# Test Main Pipeline (ChurnFeatureEngineer)
# ============================================

def test_feature_engineer_fit_transform(sample_X_y):
    """Test that ChurnFeatureEngineer fits and transforms correctly."""
    X, y = sample_X_y
    
    engineer = ChurnFeatureEngineer()
    X_transformed, y_encoded = engineer.fit_transform(X, y)
    
    # Should have more features than input (due to one-hot encoding)
    assert len(X_transformed.columns) > len(X.columns)
    
    # Target should be encoded as 0/1
    assert set(y_encoded) == {0, 1}
    
    # No missing values
    assert X_transformed.isnull().sum().sum() == 0


def test_feature_engineer_drops_specified_columns(sample_X_y):
    """Test that ChurnFeatureEngineer drops unwanted columns."""
    X, y = sample_X_y
    
    engineer = ChurnFeatureEngineer()
    X_transformed, _ = engineer.fit_transform(X, y)
    
    # Should have dropped these
    assert 'CustomerID' not in X_transformed.columns
    assert 'CallMinutes' not in X_transformed.columns
    assert 'DataUsage' not in X_transformed.columns
    assert 'ChurnProbability' not in X_transformed.columns


def test_feature_engineer_creates_engineered_features(sample_X_y):
    """Test that ChurnFeatureEngineer creates all engineered features."""
    X, y = sample_X_y
    
    engineer = ChurnFeatureEngineer()
    X_transformed, _ = engineer.fit_transform(X, y)
    
    # Check engineered features exist
    assert any('TenureBucket' in col for col in X_transformed.columns)
    assert 'PricePerService' in X_transformed.columns
    assert 'IsHighRisk' in X_transformed.columns
    assert 'ContractTenureMismatch' in X_transformed.columns
    assert 'FinancialStress' in X_transformed.columns


def test_feature_engineer_no_data_leakage(sample_X_y):
    """Test that ChurnFeatureEngineer doesn't leak test data into training."""
    X, y = sample_X_y
    
    # Split into train/test
    X_train = X.iloc[:2]
    y_train = y.iloc[:2]
    X_test = X.iloc[2:]
    y_test = y.iloc[2:]
    
    engineer = ChurnFeatureEngineer()
    
    # Fit on train only
    engineer.fit(X_train, y_train)
    
    # Transform train
    X_train_transformed, _ = engineer.transform(X_train, y_train)
    
    # Transform test (should use parameters learned from train)
    X_test_transformed, _ = engineer.transform(X_test, y_test)
    
    # Both should have same number of columns
    assert len(X_train_transformed.columns) == len(X_test_transformed.columns)
    
    # Test median should come from train data, not test data
    # (HighRiskSegment uses median from fit)
    assert engineer.feature_pipeline_.named_steps['high_risk'].monthly_charges_median_ == X_train['MonthlyCharges'].median()


def test_feature_engineer_handles_single_row(sample_X_y):
    """Test that ChurnFeatureEngineer works on single row (production inference)."""
    X, y = sample_X_y
    
    # Fit on full data
    engineer = ChurnFeatureEngineer()
    engineer.fit(X, y)
    
    # Transform single row
    X_single = X.iloc[[0]]
    y_single = y.iloc[[0]]
    
    X_transformed, y_encoded = engineer.transform(X_single, y_single)
    
    assert len(X_transformed) == 1
    assert len(y_encoded) == 1


def test_feature_engineer_save_load(sample_X_y, tmp_path):
    """Test that ChurnFeatureEngineer can be saved and loaded."""
    X, y = sample_X_y
    
    # Fit engineer
    engineer = ChurnFeatureEngineer()
    engineer.fit(X, y)
    
    # Transform data
    X_transformed_original, _ = engineer.transform(X, y)
    
    # Save
    save_path = tmp_path / "test_pipeline.pkl"
    engineer.save(str(save_path))
    
    # Load
    engineer_loaded = ChurnFeatureEngineer.load(str(save_path))
    
    # Transform with loaded pipeline
    X_transformed_loaded, _ = engineer_loaded.transform(X, y)
    
    # Should produce identical results
    pd.testing.assert_frame_equal(X_transformed_original, X_transformed_loaded)


def test_feature_engineer_all_features_numeric(sample_X_y):
    """Test that all output features are numeric (ready for ML)."""
    X, y = sample_X_y
    
    engineer = ChurnFeatureEngineer()
    X_transformed, _ = engineer.fit_transform(X, y)
    
    # All columns should be numeric
    assert X_transformed.select_dtypes(include=[np.number]).shape[1] == X_transformed.shape[1]


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])