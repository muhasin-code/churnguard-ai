# Feature Engineering Plan - ChurnGuard AI

**Date:** 2026-01-15  
**Based On:** EDA findings from `notebooks/01_eda.ipynb`

---

## Features to Keep (As-Is)

| Feature | Type | Rationale |
|---------|------|-----------|
| `Age` | Numerical | Demographic signal, minimal preprocessing needed |
| `Tenure` | Numerical | **Strong signal** - inversely correlated with churn |
| `MonthlyCharges` | Numerical | **Strong signal** - higher charges → higher churn |
| `TotalCharges` | Numerical | Captures customer lifetime value |
| `Complaints` | Numerical (count) | Dissatisfaction indicator |
| `RecentSupportTickets` | Binary | Recent issues signal imminent churn |
| `LatePayments` | Numerical (count) | Financial stress indicator |

---

## Features to Drop

| Feature | Reason |
|---------|--------|
| `CustomerID` | Identifier, no predictive value |
| `CallMinutes` | Redundant - captured in `Engagement` |
| `DataUsage` | Redundant - captured in `Engagement` |
| `ChurnProbability` | Target leakage (used to generate `Churn`) |

**Note:** Keep `Engagement` as it combines call + data usage into single metric.

---

## Categorical Features (Require Encoding)

### Low Cardinality (One-Hot Encode)

| Feature | Categories | Cardinality | Encoding Method |
|---------|-----------|-------------|-----------------|
| `Gender` | Male, Female | 2 | One-Hot → Binary |
| `ContractType` | Month-to-Month, One year, Two year | 3 | One-Hot |
| `InternetService` | DSL, Fiber, No Service | 3 | One-Hot |
| `PaymentMethod` | Credit Card, Debit Card, UPI, Cash | 4 | One-Hot |

**Total one-hot columns:** 2 + 3 + 3 + 4 = **12 new columns** (after drop-first)

### Target Variable

| Feature | Type | Encoding |
|---------|------|----------|
| `Churn` | Binary categorical | Label Encode: Yes=1, No=0 |

---

## Engineered Features (To Create)

### 1. Tenure Buckets
**Rationale:** Churn patterns differ between new customers (<6 months) and loyal customers (>2 years)

**Implementation:**
```python
pd.cut(tenure, bins=[0, 6, 12, 24, 60, 100], 
       labels=['New', 'Growing', 'Established', 'Loyal', 'Veteran'])
```

**Expected Impact:** Capture non-linear tenure-churn relationship

---

### 2. Price-to-Service Ratio
**Rationale:** Customers paying high prices relative to services may perceive poor value

**Implementation:**
```python
monthly_charges / (num_services_subscribed + 1)  # +1 to avoid division by zero
```

Where `num_services_subscribed` = count of services (internet, phone, etc.)

**Expected Impact:** Identify value-perception issues

---

### 3. High-Risk Segment (Interaction Feature)
**Rationale:** Combination of low tenure + high charges + recent tickets = extreme churn risk

**Implementation:**
```python
is_high_risk = (
    (tenure < 12) & 
    (monthly_charges > df['MonthlyCharges'].median()) & 
    (recent_support_tickets == 1)
).astype(int)
```

**Expected Impact:** Flag customers needing immediate retention efforts

---

### 4. Contract-Tenure Mismatch
**Rationale:** Month-to-month contract after long tenure suggests exit planning

**Implementation:**
```python
is_mismatch = (
    (contract_type == 'Month-to-Month') & 
    (tenure > 24)
).astype(int)
```

**Expected Impact:** Early warning for loyal customers considering leaving

---

### 5. Financial Stress Score
**Rationale:** Combine late payments + complaints as financial distress signal

**Implementation:**
```python
financial_stress = late_payments + (complaints * 0.5)
```

**Expected Impact:** Identify customers with payment difficulties

---

## Feature Scaling Strategy

**Features requiring scaling** (StandardScaler):
- `Age` (range: 18-74)
- `Tenure` (range: 0-71)
- `MonthlyCharges` (range: 10-120)
- `TotalCharges` (range: 0-8500)
- `Engagement` (range: -0.17-2.02)
- All engineered numerical features

**Why StandardScaler?**
- Tree-based models (XGBoost, Random Forest) don't need scaling
- But we'll test Logistic Regression (requires scaling)
- StandardScaler: mean=0, std=1 (preserves outliers better than MinMaxScaler)

**Features NOT requiring scaling:**
- Binary features (0/1)
- One-hot encoded features (already 0/1)
- Count features (Complaints, LatePayments) - optional

---

## Missing Value Strategy

**Current State:** All features 100% populated (fixed in data generation)

**Production Strategy** (if nulls encountered):

| Feature | Strategy | Rationale |
|---------|----------|-----------|
| `InternetService` | Fill with 'No Service' | Null means no internet subscription |
| `TotalCharges` | Fill with 0 | New customers (tenure=0) have no charges yet |
| `Age`, `Tenure` | Reject record | Critical features, missing = data quality issue |
| Numerical features | Median imputation | Robust to outliers |
| Categorical features | Mode imputation | Most frequent category |

**Implementation:** Create indicator features for imputed values to preserve signal.

---

## Implementation Order

1. **Drop unnecessary features**
2. **Handle missing values** (currently none, but add logic for production)
3. **Create engineered features**
4. **Encode categorical variables**
5. **Scale numerical features**
6. **Save feature names and transformers**

---

## Success Metrics

**After feature engineering:**
- ✅ Final feature count: ~25-30 features (from original 17)
- ✅ All features numeric (ready for ML models)
- ✅ No missing values
- ✅ No data leakage (train/test separation maintained)
- ✅ Transformers saved for production use

---

## Testing Strategy

**Unit tests to write:**
- Test feature creation functions (deterministic output)
- Test encoding produces expected columns
- Test no data leakage (fit on train, transform on test)
- Test pipeline works on single row (production inference)
- Test handling of edge cases (nulls, outliers, new categories)

---

---

## Implementation Results

**Date Completed:** 2026-04-18

### Feature Counts

| Stage | Count | Details |
|-------|-------|---------|
| Raw features | 16 | After dropping target |
| Dropped features | 4 | CustomerID, CallMinutes, DataUsage, ChurnProbability |
| Engineered features | 5 sub-features | TenureBuckets(4 bins), PricePerService, IsHighRisk, ContractMismatch, FinancialStress |
| One-hot encoded | 12 | From ContractType(2), InternetService(2), PaymentMethod(3), Gender(1), TenureBuckets(4) |
| **Final features** | **24** | Ready for model training |

**Note on feature count:** The original estimate was 28 features. The actual count is 24 because
one-hot encoding with `drop_first=True` produces k−1 columns per categorical feature (e.g. a 3-category
feature produces 2 columns, not 3). This is correct and expected behaviour — it avoids the
dummy variable trap for linear models.

### Top Features by Correlation with Churn

| Rank | Feature | Correlation | Type |
|------|---------|-------------|------|
| 1 | TenureBucket_New | 0.42 | Engineered (one-hot) |
| 2 | ContractType_Month-to-Month | 0.38 | Encoded |
| 3 | IsHighRisk | 0.35 | Engineered |
| 4 | MonthlyCharges | 0.28 | Original (scaled) |
| 5 | ContractTenureMismatch | 0.22 | Engineered |

### Data Quality Verification

✅ **Zero missing values** in train and test sets  
✅ **Zero infinite values**  
✅ **All features numeric** (int64 or float64)  
✅ **No duplicates**  
✅ **Target distribution preserved** (stratified split)

### Files Generated

- `data/processed/features_v1_train.csv` (40,000 rows, 25 columns — 24 features + target)
- `data/processed/features_v1_test.csv` (10,000 rows, 25 columns — 24 features + target)
- `models/feature_pipeline.pkl` (fitted ChurnFeatureEngineer)

### Next Iteration Ideas

**Add in v2:**
- Interaction feature: `Tenure × MonthlyCharges`
- Temporal features: `DaysSinceLastPayment`
- Polynomial features for non-linear relationships
- Feature selection based on model importance (remove low-value features)

**Experiments to run:**
- Compare StandardScaler vs RobustScaler performance
- Test target encoding for high-cardinality features (if added)
- A/B test dropping Gender (minimal correlation)