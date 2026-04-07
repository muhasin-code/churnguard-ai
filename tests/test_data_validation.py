"""
Unit tests for data validation module.

Tests both the DataValidator class and specific validation scenarios.
"""


import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.data.validation import DataValidator


@pytest.fixture
def validator():
    """
    Create a DataValidator instance for testing.
    """
    return DataValidator()


@pytest.fixture
def unit_test_suite(validator):
    """
    Create a minimal expectation suite for unit testing.
    """
    suite_name = "unit_test_suite"
    try:
        suite = validator.context.get_expectation_suite(suite_name)
        validator.context.delete_expectation_suite(suite_name)
    except:
        pass
    
    suite = validator.context.add_or_update_expectation_suite(suite_name)
    
    # Add one simple expectation that will pass
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    config = ExpectationConfiguration(
        expectation_type="expect_table_columns_to_match_ordered_list",
        kwargs={
            "column_list": [
                "CustomerID", "Gender", "Age", "Tenure", "ContractType", 
                "InternetService", "MonthlyCharges", "TotalCharges", 
                "CallMinutes", "DataUsage", "Complaints", 
                "RecentSupportTickets", "PaymentMethod", "LatePayments", 
                "Engagement", "ChurnProbability", "Churn"
            ]
        }
    )
    suite.add_expectation(config)
    
    # Add a categorical constraint that we can use for failure testing
    config_cat = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={
            "column": "Gender",
            "value_set": ["Male", "Female"]
        }
    )
    suite.add_expectation(config_cat)

    # Add a numeric constraint for failure testing
    config_num = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "CallMinutes",
            "min_value": 0,
            "max_value": 2000
        }
    )
    suite.add_expectation(config_num)
    
    validator.context.save_expectation_suite(suite)
    return suite_name


@pytest.fixture
def valid_sample_data():
    """
    Create a small valid dataset for testing.
    """
    return pd.DataFrame({
        'CustomerID': ['CUST001', 'CUST002', 'CUST003'],
        'Gender': ['Male', 'Female', 'Male'],
        'Age': [25, 30, 45],
        'Tenure': [12, 24, 6],
        'ContractType': ['Month-to-Month', 'One Year', 'Two Year'],
        'InternetService': ['Fiber', 'DSL', 'No Service'],
        'MonthlyCharges': [75.5, 65.3, 45.0],
        'TotalCharges': [906.0, 1567.2, 270.0],
        'CallMinutes': [450.0, 520.0, 380.0],
        'DataUsage': [5.2, 4.8, 0.0],
        'Complaints': [0, 1, 0],
        'RecentSupportTickets': [0, 1, 0],
        'PaymentMethod': ['Credit Card', 'UPI', 'Cash'],
        'LatePayments': [0, 0, 1],
        'Engagement': [0.95, 1.02, 0.38],
        'ChurnProbability': [0.35, 0.42, 0.68],
        'Churn': ['No', 'No', 'Yes']
    })


@pytest.fixture
def invalid_data_negative_usage():
    """
    Create dataset with negative usage values.
    """
    df = pd.DataFrame({
        'CustomerID': ['CUST001'],
        'Gender': ['Male'],
        'Age': [25],
        'Tenure': [12],
        'ContractType': ['Month-to-Month'],
        'InternetService': ['Fiber'],
        'MonthlyCharges': [75.5],
        'TotalCharges': [906.0],
        'CallMinutes': [-50.0],  # INVALID!
        'DataUsage': [5.2],
        'Complaints': [0],
        'RecentSupportTickets': [0],
        'PaymentMethod': ['Credit Card'],
        'LatePayments': [0],
        'Engagement': [0.95],
        'ChurnProbability': [0.35],
        'Churn': ['No']
    })

    return df


@pytest.fixture
def invalid_data_wrong_category():
    """
    Create dataset with invalid categorical value.
    """
    df = pd.DataFrame({
        'CustomerID': ['CUST001'],
        'Gender': ['Other'],  # INVALID! Not in ['Male', 'Female']
        'Age': [25],
        'Tenure': [12],
        'ContractType': ['Month-to-Month'],
        'InternetService': ['Fiber'],
        'MonthlyCharges': [75.5],
        'TotalCharges': [906.0],
        'CallMinutes': [450.0],
        'DataUsage': [5.2],
        'Complaints': [0],
        'RecentSupportTickets': [0],
        'PaymentMethod': ['Credit Card'],
        'LatePayments': [0],
        'Engagement': [0.95],
        'ChurnProbability': [0.35],
        'Churn': ['No']
    })

    return df


def test_validator_initialization():
    """
    Test that DataValidator initializes correctly.
    """
    validator = DataValidator()
    assert validator.context is not None
    assert hasattr(validator, 'validate_raw_data')


def test_valid_data_passes(validator, valid_sample_data, unit_test_suite):
    """
    Test that valid data passes validation.
    """
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        valid_sample_data.to_csv(temp_path, index=False)
    
    try:
        # Should not raise exception
        result = validator.validate_raw_data(temp_path, expectation_suite_name=unit_test_suite)
        assert result == True
    finally:
        Path(temp_path).unlink()


def test_negative_usage_fails(validator, invalid_data_negative_usage, unit_test_suite):
    """
    Test that negative usage values are caught.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        invalid_data_negative_usage.to_csv(temp_path, index=False)
    
    try:
        with pytest.raises(ValueError, match="Data validation failed"):
            validator.validate_raw_data(temp_path, expectation_suite_name=unit_test_suite)
    finally:
        Path(temp_path).unlink()


def test_wrong_category_fails(validator, invalid_data_wrong_category, unit_test_suite):
    """Test that invalid categorical values are caught."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        invalid_data_wrong_category.to_csv(temp_path, index=False)
    
    try:
        with pytest.raises(ValueError, match="Data validation failed"):
            validator.validate_raw_data(temp_path, expectation_suite_name=unit_test_suite)
    finally:
        Path(temp_path).unlink()


def test_missing_file_raises_error(validator):
    """Test that missing file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        validator.validate_raw_data("nonexistent_file.csv")


def test_validation_with_actual_dataset(validator):
    """Test validation with actual generated dataset."""
    # Only run if actual file exists
    data_path = "data/raw/telecom_data.csv"
    
    if Path(data_path).exists():
        result = validator.validate_raw_data(data_path)
        assert result == True
    else:
        pytest.skip("Actual dataset not found")