"""
Create Great Expectations suite for raw churn data.

This script programmatically defines all data quality expectations
instead of using the interactive notebook workflow.
"""

import great_expectations as gx
from great_expectations.core.batch import BatchRequest

# Get context
context = gx.get_context()

# Create expectation suite
suite_name = "churn_raw_data_expectations"

# Remove existing suite if it exists
try:
    context.delete_expectation_suite(suite_name)
    print(f"Deleted existing suite: {suite_name}")
except:
    pass

# Create new suite
suite = context.add_expectation_suite(suite_name)
print(f"Created new suite: {suite_name}")

# Create batch request
batch_request = {
    'datasource_name': 'churn_data_source',
    'data_connector_name': 'default_inferred_data_connector_name',
    'data_asset_name': 'telecom_data',
}

# Get validator
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name=suite_name
)

print(f"Loaded data: {len(validator.active_batch.data)} rows")

# Add all expectations (same as in the detailed guide)
# ... (I'll provide the full code in next message)

# Save suite
validator.save_expectation_suite(discard_failed_expectations=False)
print(f"Saved suite with {len(suite.expectations)} expectations")