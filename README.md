# ChurnGuard AI

> Production-grade customer churn prediction system with full MLOps pipeline

## Problem Statement
Customer churn costs SaaS companies millions annually. ChurnGuard AI predicts which customers are likely to churn, enabling proactive retention campaigns. The system is built with a full MLOps pipeline: data versioning (DVC), experiment tracking (MLflow), automated data validation (Great Expectations), and interpretability (SHAP).

## Business Impact
- **Early detection:** Identify at-risk customers before they churn
- **Cost savings:** Champion model saves ~$44,850/year vs. baseline through better churn detection and fewer false alarms
- **Efficiency:** Automated pipeline reduces manual analysis time by 90%

## System Architecture
```
data/raw/ ──► Great Expectations ──► src/features/ ──► MLflow ──► models/
   │              (validation)        (engineering)   (tracking)     │
   │                                                                  │
   └─── DVC (versioning) ────────────────────────────────────────────┘
                                                                      │
                                                               FastAPI (Phase 3)
```

## Tech Stack
- **ML Framework:** scikit-learn, XGBoost, LightGBM
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Data Validation:** Great Expectations
- **API:** FastAPI (Phase 3 — planned)
- **Monitoring:** Evidently (Phase 4 — planned)

---

## Data Generation

**Dataset:** `data/raw/telecom_data.csv`
- **Rows:** 50,000 synthetic customer records
- **Churn Rate:** ~65.2%
- **Generation:** `scripts/generate_synthetic_data_v3.1.py`

### Version History

**v1.0 (Deprecated — Buggy)**
- ❌ Negative usage values from unbounded distributions
- ❌ InternetService nulls ambiguous
- ❌ Inconsistent dtypes

**v2.0 (Deprecated — Data Leakage)**
- ✅ Fixed: Clipped distributions (CallMinutes, DataUsage ≥ 0)
- ✅ Fixed: InternetService "No Service" explicit category
- ✅ Fixed: Enforced dtypes (int64, float64, category)
- ❌ `RecentSupportTickets` coefficient (1.5) too high — caused dominant feature signal

**v3.0 (Deprecated — Under-powered)**
- ✅ Fixed: RecentSupportTickets coefficient reduced (1.5 → 0.3) — eliminated leakage
- ❌ Result: ROC-AUC only ~0.71 (signal too weak)

**v3.1 (Current)**
- ✅ Optimised all coefficients for 0.80+ ROC-AUC
- ✅ Added high-charges rule and very-low-engagement rule for better separation
- ✅ Validated: RecentSupportTickets is rank 7 in SHAP importance (no leakage)
- ✅ Result: ROC-AUC 0.8077 with balanced feature importance

### Regenerating Data
```bash
# Default (50K rows, seed=42)
python scripts/generate_synthetic_data_v3.1.py

# Custom size
python scripts/generate_synthetic_data_v3.1.py --rows 100000

# Different seed
python scripts/generate_synthetic_data_v3.1.py --seed 123
```

---

## Data Quality

All data undergoes automated validation using Great Expectations:

- ✅ 42 data quality expectations
- ✅ Schema validation (17 columns, correct types)
- ✅ Range validation (no negative usage values)
- ✅ Categorical validation (valid enum values)
- ✅ Statistical validation (churn rate stability: 60–70%)

**View data contract:** [docs/data_contract.md](docs/data_contract.md)

---

## Feature Engineering

**24 features** engineered from 16 raw features (after dropping 4 non-predictive columns):

### Engineered Features
- **Tenure Buckets:** Customer lifecycle stages (New → Veteran) — one-hot encoded
- **Price-to-Service Ratio:** Value perception metric
- **High-Risk Segment:** Multi-factor churn risk flag (low tenure + high charges + tickets)
- **Contract-Tenure Mismatch:** Exit planning detection (month-to-month after long tenure)
- **Financial Stress Score:** Combined late payments + complaints indicator

### Encoding & Scaling
- One-hot encoding for categorical variables (`drop_first=True`)
- StandardScaler for numerical features (mean=0, std=1)
- Binary label encoding for target (Churn: Yes=1, No=0)

### Reproducibility
All transformations saved in `models/feature_pipeline.pkl` for production use.

**View details:** [Feature Engineering Plan](docs/feature_engineering_plan.md)

---

## Model Development

### Baseline Model (Logistic Regression)

**Performance (Actual):**
- **ROC-AUC:** 0.7944
- **Accuracy:** 72.66%
- **Precision:** 82.99%
- **Recall:** 73.04%

**Status:** ✅ Baseline established (exceeds minimum 0.75 ROC-AUC threshold)

**Key Findings:**
- Long-term contracts (1-year, 2-year) strongly reduce churn
- Recent support tickets and complaints are major churn indicators
- Model captures ~73% of churners with high precision (~83%)

**View full model card:** [docs/model_cards/baseline_logistic_regression.md](docs/model_cards/baseline_logistic_regression.md)

**MLflow Tracking:** All experiments tracked at http://localhost:5000

---

## Model Experimentation Results

**Experiments:** 11 models trained and compared in MLflow
**Champion:** XGBoost Conservative (ROC-AUC: **0.8077**)
**Business Value:** ~$44,850 annual savings from improved predictions vs. baseline

### Model Comparison (Actual Results)

| Model Type | Best ROC-AUC | Status |
|------------|--------------|--------|
| XGBoost | **0.8077** | ✅ Champion |
| Stacking Ensemble | 0.8075 | Ties champion, 4× complexity |
| LightGBM | 0.8060 | Runner-up |
| Voting Ensemble | 0.8057 | Weaker than single model |
| Random Forest | 0.7958 | Good recall, lower AUC |
| Baseline (Logistic) | 0.7944 | Benchmark |

**Key Findings:**
- Gradient boosting significantly outperforms linear baseline (+1.7%)
- Conservative regularisation beats aggressive configurations
- Ensembles add near-zero value due to high model correlation
- Champion selected based on performance + simplicity trade-off

**View detailed analysis:** [Model Experimentation Report](docs/model_experimentation_report.md)

---

## Model Evaluation & Interpretability (Milestone 2.3)

**Champion model evaluated with SHAP** on 10,000 holdout customers.

### SHAP Feature Importance (Top 10)

| Rank | Feature | Mean |SHAP| |
|------|---------|------|
| 1 | ContractType_One Year | 0.4701 |
| 2 | ContractType_Two Year | 0.4331 |
| 3 | Tenure | 0.3916 |
| 4 | MonthlyCharges | 0.3752 |
| 5 | Engagement | 0.2814 |
| 6 | FinancialStress | 0.2439 |
| 7 | RecentSupportTickets | 0.2277 |
| 8 | Complaints | 0.2000 |

**Contract type is the dominant predictor** — customers without a long-term commitment are highest risk.
All engineered features (`FinancialStress`, `ContractTenureMismatch`) contribute meaningfully.

### Threshold Optimisation

The cost-based optimiser found threshold=0.10 minimises cost given $50/FP and $600/FN.
However, this is operationally impractical (flags ~33% of all customers).
**Practical recommendation: use threshold 0.35–0.45** pending business stakeholder input.

**View full evaluation:** [Model Evaluation Report](docs/model_evaluation_report.md)

All plots saved to `evaluation_results/`:
- ROC curve, PR curve, calibration curve
- Confusion matrices (raw + normalised)
- SHAP summary, waterfall, and dependence plots
- Error analysis CSV and distribution plot

---

## Project Status

🔄 **In Progress** — Phase II: Model Development & Experiment Tracking

## Milestones
- [x] Project Setup (environment, DVC, Great Expectations)
- [x] Data Pipeline (synthetic data v3.1, feature engineering)
- [x] MLflow Setup & Baseline Model
- [x] Model Experimentation (11 models, champion selected)
- [x] Model Evaluation & Interpretability (SHAP, error analysis, calibration)
- [ ] **Model Registry Setup** ← Next
- [ ] API Development (FastAPI)
- [ ] Deployment (Docker)
- [ ] Production Monitoring (Evidently)

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd churnguard-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python scripts/generate_synthetic_data_v3.1.py

# 3. Run feature engineering
python scripts/engineer_features.py

# 4. Train models
python src/models/train_models.py

# 5. Evaluate champion
python scripts/save_champion_from_mlflow.py
python scripts/evaluate_champion.py
```

---

## Author
Muhammed Muhasin K
LinkedIn — https://www.linkedin.com/in/muhasin-code

## License
MIT License