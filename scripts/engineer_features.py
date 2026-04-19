"""
Feature engineering script for ChurnGuard AI.

Transforms raw data into ML-ready features.

Usage:
    python scripts/engineer_features.py --input data/raw/telecom_data.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import ChurnFeatureEngineer


def main():
    parser = argparse.ArgumentParser(
        description="Engineer features for churn prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process raw data and save features
  python scripts/engineer_features.py
  
  # Specify custom input/output paths
  python scripts/engineer_features.py \\
    --input data/raw/telecom_data.csv \\
    --output data/processed/features_v1.csv
  
  # Save the fitted pipeline for later use
  python scripts/engineer_features.py --save-pipeline models/feature_pipeline.pkl
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/telecom_data.csv',
        help='Path to raw data CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/features_v1.csv',
        help='Path to save processed features'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/feature_config.yaml',
        help='Path to feature config YAML'
    )
    
    parser.add_argument(
        '--save-pipeline',
        type=str,
        default='models/feature_pipeline.pkl',
        help='Path to save fitted feature pipeline'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for train/test split'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("=" * 60)
    print("ChurnGuard AI - Feature Engineering")
    print("=" * 60)
    print(f"\n  Input:    {args.input}")
    print(f"  Output:   {args.output}")
    print(f"  Config:   {args.config}")
    print(f"  Pipeline: {args.save_pipeline}\n")
    
    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv(args.input)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Separate features and target
    if 'Churn' not in df.columns:
        print("Error: 'Churn' column not found in data")
        sys.exit(1)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    print(f"   Features: {len(X.columns)} columns")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    
    print(f"\nSplitting data (test_size={args.test_split})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_split,
        random_state=args.random_state,
        stratify=y  # Maintain churn ratio in both sets
    )
    
    print(f"   Train: {len(X_train):,} rows")
    print(f"   Test:  {len(X_test):,} rows")
    
    # Initialize feature engineer
    print(f"\nInitializing feature engineer...")
    engineer = ChurnFeatureEngineer(config_path=args.config)
    
    # Fit on training data
    print("Fitting transformations on training data...")
    engineer.fit(X_train, y_train)
    
    # Transform both sets
    print("Transforming training data...")
    X_train_transformed, y_train_encoded = engineer.transform(X_train, y_train)
    
    print("Transforming test data...")
    X_test_transformed, y_test_encoded = engineer.transform(X_test, y_test)
    
    print(f"\nTransformation complete!")
    print(f"   Input features:  {len(X.columns)}")
    print(f"   Output features: {len(X_train_transformed.columns)}")
    
    if args.verbose:
        print(f"\nOutput features:")
        for i, col in enumerate(X_train_transformed.columns, 1):
            print(f"   {i:2d}. {col}")
    
    # Combine train and test back together with target
    print(f"\nSaving processed features...")
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Save training set
    train_output = Path(args.output).parent / f"{Path(args.output).stem}_train.csv"
    train_df = X_train_transformed.copy()
    train_df['Churn'] = y_train_encoded
    train_df.to_csv(train_output, index=False)
    print(f"   Train: {train_output} ({len(train_df):,} rows)")
    
    # Save test set
    test_output = Path(args.output).parent / f"{Path(args.output).stem}_test.csv"
    test_df = X_test_transformed.copy()
    test_df['Churn'] = y_test_encoded
    test_df.to_csv(test_output, index=False)
    print(f"   Test:  {test_output} ({len(test_df):,} rows)")
    
    # Save feature pipeline
    print(f"\nSaving feature pipeline...")
    engineer.save(args.save_pipeline)
    
    # Generate feature summary
    print(f"\nFeature Engineering Summary:")
    print(f"   {'='*50}")
    print(f"   Original features:     {len(X.columns)}")
    print(f"   Dropped features:      {len(engineer.config['features_to_drop'])}")
    print(f"   Engineered features:   {len(X_train_transformed.columns) - len(X.columns) + len(engineer.config['features_to_drop'])}")
    print(f"   Final feature count:   {len(X_train_transformed.columns)}")
    print(f"   {'='*50}")
    
    # Data quality checks
    print(f"\nData Quality Checks:")
    print(f"   Missing values (train): {X_train_transformed.isnull().sum().sum()}")
    print(f"   Missing values (test):  {X_test_transformed.isnull().sum().sum()}")
    print(f"   Infinite values (train): {np.isinf(X_train_transformed.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"   Infinite values (test):  {np.isinf(X_test_transformed.select_dtypes(include=[np.number])).sum().sum()}")
    
    print(f"\nFeature engineering complete!")
    print(f"\nNext steps:")
    print(f"   1. Review output features in {train_output}")
    print(f"   2. Track processed data with DVC:")
    print(f"      dvc add {train_output}")
    print(f"      dvc add {test_output}")
    print(f"   3. Start model training with processed features")


if __name__ == "__main__":
    main()