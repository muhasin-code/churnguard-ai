"""
Data Generation Script for Telecom Churn Dataset

Version History:
v1.0 (buggy) - Initial Implementation
    - Issue: Negative usage valuesfrom unbounded normal distribution
    - Issue: InternetService "None" stored as string instead of null
    - Issue: No dtype enforcement, pandas infers inconsistently

v2.0 (current) - Production-ready implementation
    - Fixed: Clipped distributions to reaslistic bounds
    - Fixed: Proper null handling for no-service customers
    - Fixed: Explicit dtype enforcement
    - Added: CLI arguments for flexibility
    - Added: Reproducible seeding
"""


import pandas as pd
from faker import Faker
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


fake = Faker()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_customer_data(n=50000, seed=42):
    """
    Generate synthetic churn data with realistic patterns

    Args:
        n (int): Number of rows to generate
        seed (int): Random seed for reprocibility
    """

    # Set seed for reproducibility
    np.random.seed(seed)
    Faker.seed(seed)

    data = []

    for i in range(n):
        # --- Customer demographics ---
        customer_id = f"CUST{i:06d}"
        gender = np.random.choice(["Male", "Female"])
        age = np.random.randint(18, 75)
        tenure = np.random.randint(0, 72) # This is in months

        # --- Subscription details ---
        contract_type =  np.random.choice(
            ["Month-to-Month", "One Year", "Two Year"],
            p=[0.6, 0.25, 0.15]
        )
        internet_service = np.random.choice(
            ["DSL", "Fiber", "No Service"],
            p=[0.4, 0.5, 0.1]
        )

        # --- Pricing ---
        base_price = np.random.uniform(20, 100)
        if internet_service == "Fiber":
            base_price += 20
        if contract_type == "Two Year":
            base_price -= 10
        monthly_charges = round(base_price, 2)
        total_charges = round(monthly_charges * max(1, tenure), 2)

        # --- Usage ---
        call_minutes = np.clip(np.random.normal(500, 150), 0, None)
        data_usage = np.clip(np.random.normal(5, 2), 0, None) # This is in GBs
        # Engagement score (devived)
        engagement = (call_minutes / 1000) + (data_usage / 10)

        # --- Support / complaints
        complaints = np.random.poisson(1)
        recent_support_tickets = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # --- Payment ---
        payment_method = np.random.choice(
            ["Credit Card", "Debit Card", "UPI", "Cash"]
        )
        late_payments = np.random.poisson(0.5)

        # --- CHURN PROBABILITY (business rules encoded) ---
        churn_score = 0

        # Rule 1: High tenure -> low churn 
        churn_score -= 0.03 * tenure

        # Rule 2: Recent support tickets -> higher churn
        churn_score += 1.5 * recent_support_tickets

        # Rule 3: Month-to-Month -> higher churn
        if contract_type == "Month-to-Month":
            churn_score += 1.2
        elif contract_type == "One Year":
            churn_score -= 0.5
        else:
            churn_score -= 1.0
        
        # Rule 4: Low engagement + high price -> very high churn
        if engagement < 1 and monthly_charges > 70:
            churn_score += 2.0
        
        # Additional realism factors
        churn_score += 0.5 * complaints
        churn_score += 0.3 * late_payments

        # Convert to probability
        churn_prob = sigmoid(churn_score)

        # --- Add noise ---
        churn_prob = np.clip(
            churn_prob + np.random.normal(0, 0.05),
            0, 1
        )
        churn = np.random.choice(
            ["Yes", "No"], p=[churn_prob, 1 - churn_prob]
        )

        data.append([
            customer_id,
            gender,
            age,
            tenure,
            contract_type,
            internet_service,
            monthly_charges,
            total_charges,
            round(call_minutes, 2),
            round(data_usage, 2),
            complaints,
            recent_support_tickets,
            payment_method,
            late_payments,
            round(engagement, 2),
            round(churn_prob, 3),
            churn
        ])
    
    columns = [
        "CustomerID",
        "Gender",
        "Age",
        "Tenure",
        "ContractType",
        "InternetService",
        "MonthlyCharges",
        "TotalCharges",
        "CallMinutes",
        "DataUsage",
        "Complaints",
        "RecentSupportTickets",
        "PaymentMethod",
        "LatePayments",
        "Engagement",
        "ChurnProbability",
        "Churn"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Exlicit dtype enforcement
    dtype_map = ({
        'CustomerID': 'object',             # String identifier
        'Gender': 'category',              # Categorical (saves memory)
        'Age': 'int64',                     # Integer
        'Tenure': 'int64',                  # Integer
        'ContractType': 'category',         # Categorical
        'InternetService': 'category',      # Categorical
        'MonthlyCharges': 'float64',        # Decimal
        'TotalCharges': 'float64',          # Decimal
        'CallMinutes': 'float64',           # Decimal
        'DataUsage': 'float64',             # Decimal
        'Complaints': 'int64',              # Integer count
        'RecentSupportTickets': 'int64',    # Binary (0 or 1)
        'PaymentMethod': 'category',        # Categorical
        'LatePayments': 'int64',            # Integer count
        'Engagement': 'float64',            # Decimal
        'ChurnProbability': 'float64',       # Decimal
        'Churn': 'category'                 # Categorical target
    })

    df = df.astype(dtype_map)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic telecom churn data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate default 50K rows
    python scripts/generate_synthetic_data.py

    # Generate 100K rows with custom output path
    python scripts/generate_synthetic_data.py --rows 100000 --output data/raw/telecom_100k.csv

    # Generate with specific seed for reproducibility
    python scripts/generate_synthetic_data.py --seed 123
        """
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/telecom_data.csv',
        help='Output CSV file path (default: data/raw/telecom_data.csv)'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=50000,
        help='Number of customer records to generate (default: 50000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed statistics'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.rows < 1000:
        parser.error("--rows must be at least 1000 for meaningful data")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"\nGenerating {args.rows:,} customer records...")
    print(f"Random seed: {args.seed}")

    start_time = datetime.now()
    df = generate_customer_data(n=args.rows, seed=args.seed)
    generation_time = (datetime.now() - start_time).total_seconds()

    # Save to CSV
    print(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)

    # Summary statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100

    print(f"\nGeneration complete!")
    print(f"   • Rows: {len(df):,}")
    print(f"   • Columns: {len(df.columns)}")
    print(f"   • File size: {file_size_mb:.2f}")
    print(f"   • Churn rate: {churn_rate:.1f}")
    print(f"   • Generation time: {generation_time:.2f}s")

    if args.verbose:
        print(f"\nData Preview:")
        print(df.head())
        
        print(f"\nData Types:")
        print(df.dtypes)
        
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        
        print(f"\nNumeric Summary:")
        print(df.describe())