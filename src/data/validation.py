"""
Data validation module using Great Expectations.

Provides validation for raw and processed data in the ChurnGuard AI pipeline.
"""

import great_expectations as gx
from great_expectations.core.batch import BatchRequest
from pathlib import Path
import sys
import argparse


class DataValidator:
    """
    Validates data against Great Expectations suites.
    
    Example:
        validator = DataValidator()
        validator.validate_raw_data('data/raw/telecom_data.csv')
    """
    
    def __init__(self, context_root_dir: str = "gx"):
        """
        Initialize validator with GX context.
        
        Args:
            context_root_dir: Path to Great Expectations directory
        """
        self.context = gx.get_context(context_root_dir=context_root_dir)
    
    def validate_raw_data(self, filepath: str) -> bool:
        """
        Validate raw telecom churn data.
        
        Args:
            filepath: Path to CSV file to validate
            
        Returns:
            True if validation passes, raises exception otherwise
            
        Raises:
            ValueError: If validation fails with details of failures
            FileNotFoundError: If the data file does not exist
        """
        print(f"Validating raw data: {filepath}")
        
        # Check file exists
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Fix: Use correct BatchRequest signature for GX < 1.0
        # 'options' is not a valid parameter — use batch_spec_passthrough
        # and include data_connector_name which was also missing
        batch_request = BatchRequest(
            datasource_name="churn_data_source",
            data_connector_name="default_inferred_data_connector_name",
            data_asset_name="telecom_data",
            batch_spec_passthrough={
                "reader_method": "read_csv",
                "reader_options": {}
            }
        )
        
        # Run checkpoint
        checkpoint = self.context.get_checkpoint("raw_data_checkpoint")
        
        results = checkpoint.run(
            batch_request=batch_request,
            run_name=f"validation_{Path(filepath).stem}"
        )

        success = results.success

        run_result = list(results.run_results.values())[0]

        if isinstance(run_result, dict):
            validation_result = run_result["validation_result"]
        else:
            validation_result = run_result.validation_result

        statistics = validation_result["statistics"]

        if success:
            print("Validation passed!")
            print(f"   • Expectations run: {statistics['evaluated_expectations']}")
            print(f"   • Successful: {statistics['successful_expectations']}")
            print(f"   • Success rate: {statistics['success_percent']:.1f}%")
            return True
        else:
            print("Validation failed!")

            failed_expectations = [
                exp for exp in validation_result["results"]
                if not exp["success"]
            ]
            
            print(f"\nFailed Expectations ({len(failed_expectations)}):")
            for i, exp in enumerate(failed_expectations[:5], 1):  # Show first 5
                exp_type = exp["expectation_config"]["expectation_type"]
                kwargs = exp["expectation_config"]["kwargs"]
                print(f"   {i}. {exp_type}")
                print(f"      Column: {kwargs.get('column', 'N/A')}")
                print(f"      Details: {exp.get('result', {})}")
            
            if len(failed_expectations) > 5:
                print(f"   ... and {len(failed_expectations) - 5} more")
            
            # Raise exception with details
            raise ValueError(
                f"Data validation failed. "
                f"{len(failed_expectations)} expectations failed. "
                f"Check Data Docs for details."
            )
    
    def generate_data_docs(self):
        """Generate and open Data Docs HTML report."""
        print("Generating Data Docs...")
        self.context.build_data_docs()
        self.context.open_data_docs()
        print("Data Docs generated!")


# CLI for standalone usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ChurnGuard AI data")
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/telecom_data.csv',
        help='Path to data file to validate'
    )
    parser.add_argument(
        '--docs',
        action='store_true',
        help='Generate Data Docs report'
    )
    
    args = parser.parse_args()
    
    validator = DataValidator()
    
    try:
        validator.validate_raw_data(args.input)
        
        if args.docs:
            validator.generate_data_docs()
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)