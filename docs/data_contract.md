# Data Contract: Telecom Customer Churn Dataset

**Version:** 2.0  
**Last Updated:** 2026-04-19  
**Owner:** ChurnGuard AI Team

---

## Purpose

This document defines the **data contract** for raw customer churn data entering the ChurnGuard AI system. All data must conform to these specifications to ensure pipeline reliability.

## Data Source

- **Origin:** Synthetic generation script (`scripts/generate_synthetic_data.py`)
- **Update Frequency:** On-demand (for development)
- **Production Equivalent:** Daily batch from CRM system at 2 AM UTC
- **Format:** CSV (UTF-8 encoding)
- **Delivery:** Placed in `data/raw/` directory

---

## Schema Definition

### Required Columns

| Column | Type | Nullable | Description | Constraints |
|--------|------|----------|-------------|-------------|
| `CustomerID` | string | No | Unique customer identifier | UNIQUE, NOT NULL |
| `Gender` | category | No | Customer gender | IN ('Male', 'Female') |
| `Age` | int64 | No | Customer age in years | 18 ≤ age ≤ 100 |
| `Tenure` | int64 | No | Months as customer | ≥ 0 |
| `ContractType` | category | No | Contract commitment | IN ('Month-to-Month', 'One Year', 'Two Year') |
| `InternetService` | category | No | Internet service type | IN ('DSL', 'Fiber', 'No Service') |
| `MonthlyCharges` | float64 | No | Current monthly bill (USD) | 0 < value < 500 |
| `TotalCharges` | float64 | No | Lifetime charges (USD) | ≥ 0 |
| `CallMinutes` | float64 | No | Monthly call usage (minutes) | ≥ 0 |
| `DataUsage` | float64 | No | Monthly data usage (GB) | ≥ 0 |
| `Complaints` | int64 | No | Number of complaints filed | ≥ 0 |
| `RecentSupportTickets` | int64 | No | Support ticket in last 30 days | IN (0, 1) |
| `PaymentMethod` | category | No | Payment method | IN ('Credit Card', 'Debit Card', 'UPI', 'Cash') |
| `LatePayments` | int64 | No | Count of late payments | ≥ 0 |
| `Engagement` | float64 | No | Engagement score | ≥ 0 |
| `ChurnProbability` | float64 | No | Predicted churn probability | 0 ≤ value ≤ 1 |
| `Churn` | category | No | **Target:** Did customer churn? | IN ('Yes', 'No') |

**Total Columns:** 17  
**Expected Row Count:** 45,000 - 55,000 (±10% tolerance)

---

## Data Quality Rules

### Critical Rules (MUST pass - pipeline stops if failed)

1. **Schema Completeness**
   - All 17 columns must be present
   - Column order must match specification
   - No extra columns allowed

2. **Uniqueness**
   - `CustomerID` must be unique (no duplicates)

3. **No Negative Values**
   - `CallMinutes` ≥ 0
   - `DataUsage` ≥ 0
   - `Age` ≥ 18
   - `Tenure` ≥ 0
   - `MonthlyCharges` > 0

4. **Data Type Compliance**
   - Numeric columns must be parseable as int64 or float64
   - Categorical columns must contain only specified values
   - No type mismatches allowed

5. **Null Policy**
   - **Zero tolerance:** All columns must be 100% populated
   - Missing data is a critical failure

### Warning Rules (SHOULD pass - logged but pipeline continues)

1. **Churn Rate Stability**
   - Churn rate should be between 60-70%
   - Deviation triggers investigation

2. **Row Count Stability**
   - Expected: ~50,000 rows
   - Alert if <45,000 or >55,000

### Informational Rules (For monitoring only)

1. **Feature Distribution**
   - Track mean/std of numeric features
   - Track category frequencies
   - Detect data drift over time

---

## Validation Process

### Automated Validation

All incoming data is validated using **Great Expectations** before entering the pipeline.

**Validation Checkpoint:** `raw_data_checkpoint`  
**Expectation Suite:** `churn_raw_data_expectations`  
**Total Expectations:** 42

### Validation Workflow
                  ┌─────────────────┐
                  │  New Data File  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Run GX Check   │  ← Validates against all 42 expectations
                  └────────┬────────┘
                           │
                     ┌───┴───┐
                     │       │
                     ▼       ▼
                  ┌──────┐  ┌──────┐
                  │ PASS │  │ FAIL │
                  └───┬──┘  └───┬──┘
                     │         │
                     │         ▼
                     │    ┌────────────────────┐
                     │    │ Alert Team         │
                     │    │ Generate Report    │
                     │    │ Block Pipeline     │
                     │    └────────────────────┘
                     │
                     ▼
                  ┌─────────────────┐
                  │  Proceed to     │
                  │  Feature Eng.   │
                  └─────────────────┘

### Running Validation Manually
```bash
# From Python
python src/data/validation.py --input data/raw/telecom_data.csv

# Using Great Expectations CLI
great_expectations checkpoint run raw_data_checkpoint

# Generate validation report
great_expectations docs build
```

---

## Change Management

### Adding New Columns

1. Update this contract document
2. Add corresponding Great Expectations
3. Update feature engineering pipeline (`src/features/build_features.py`)
4. Update unit tests
5. Increment contract version
6. Communicate changes to data producers

### Modifying Constraints

1. Analyze 30 days of historical data
2. Document rationale (link to analysis notebook)
3. Update expectations in GX
4. Re-run validation on historical data
5. Update this contract
6. Version change in Git

### Deprecating Columns

1. Mark as deprecated in contract (6-month notice)
2. Update downstream code to handle absence
3. Remove from expectations
4. Remove from contract

---

## Violation Handling

### Critical Violations

**Examples:**
- Negative `CallMinutes`
- Missing `CustomerID`
- Wrong data types

**Response:**
1. Pipeline stops immediately
2. Slack/email alert sent to team
3. Data Docs report generated
4. On-call engineer investigates
5. Data re-generated or fixed at source
6. Manual approval required to resume

### Warning Violations

**Examples:**
- Churn rate = 75% (expected 60-70%)
- Row count = 40,000 (expected ~50,000)

**Response:**
1. Pipeline continues
2. Warning logged
3. Daily summary report
4. Investigation within 24 hours

---

## Contact & Escalation

- **Data Quality Issues:** [your-email@example.com]
- **Schema Change Requests:** [team-lead@example.com]
- **On-Call (Critical Failures):** [pagerduty-link]

---

## Appendix: Example Valid Record
```json
{
  "CustomerID": "CUST000042",
  "Gender": "Female",
  "Age": 32,
  "Tenure": 18,
  "ContractType": "One Year",
  "InternetService": "Fiber",
  "MonthlyCharges": 78.50,
  "TotalCharges": 1413.00,
  "CallMinutes": 520.30,
  "DataUsage": 6.2,
  "Complaints": 1,
  "RecentSupportTickets": 0,
  "PaymentMethod": "Credit Card",
  "LatePayments": 0,
  "Engagement": 1.04,
  "ChurnProbability": 0.42,
  "Churn": "No"
}
```

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-01 | Initial contract | You |
| 2.0 | 2026-01-15 | Fixed negative values, enforced dtypes, added GX validation | You |
| 2.1 | 2026-04-19 | Fixed ContractType casing ('One year' → 'One Year', 'Two year' → 'Two Year') to match v3.1 data generation | Muhasin |