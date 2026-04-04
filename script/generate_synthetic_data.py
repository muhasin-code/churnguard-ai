import pandas as pd
from faker import Faker
import numpy as np
import argparse
from pathlib import Path


fake = Faker()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_customer_data(n=50000):
    """
    Generate synthetic churn data with realistic patterns
    """
    data = []

    for i in range(n):
        # --- Customer demographics ---
        customer_id = f"CUST{i:06d}"
        gender = np.random.choice(["Male", "Female"])
        age = np.random.randint(18, 75)
        tenure = np.random.randint(0, 72) # This is in months

        # --- Subscription details ---
        contract_type =  np.random.choice(
            ["Month-to-Month", "One year", "Two year"],
            p=[0.6, 0.25, 0.15]
        )
        internet_service = np.random.choice(
            ["DSL", "Fiber", np.nan],
            p=[0.4, 0.5, 0.1]
        )

        # --- Pricing ---
        base_price = np.random.uniform(20, 100)
        if internet_service == "Fiber":
            base_price += 20
        if contract_type == "Two year":
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
        elif contract_type == "One year":
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

    df = df.astype({
        'Age': 'int64',
        'Tenure': 'int64',
        'MonthlyCharges': 'float64',
        'TotalCharges': 'float64',
        'CallMinutes': 'float64',
        'DataUsage': 'float64',
        'Complaints': 'int64',
        'RecentSupportTickets': 'int64',
        'LatePayments': 'int64',
        'Engagement': 'float64',
        'ChurnProbability': 'float64'
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic telecom churn data")
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/telecom_data.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=50000,
        help='Number of rows to generate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Generate data
    print(f"Generating {args.rows} rows...")
    df = generate_customer_data(n=args.rows)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Save
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")