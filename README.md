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

## Model Registry & Deployment

**Current Production Model:** churnguard-classifier v1  
**Performance:** ROC-AUC 0.8077 | Accuracy 73.66% | Precision 83.60%  
**Deployment Stage:** Production  
**Last Updated:** 2026-04-18

### Registry Workflow

```
Train → Register → Validate (Staging) → Deploy (Production)
```

**Registry contains:**
- ✅ Version control for models
- ✅ Performance metrics tracking
- ✅ Deployment stage management (Staging/Production)
- ✅ Model lineage and metadata
- ✅ Rollback capability

**View in MLflow UI:** http://localhost:5000/#/models/churnguard-classifier

### Quick Commands

```bash
# Register champion model
python scripts/register_champion.py

# Validate staging model
python scripts/validate_staging_model.py

# Promote to production
python scripts/promote_to_production.py

# Load production model
from src.models.registry_utils import ModelRegistry
registry = ModelRegistry()
model = registry.load_model("churnguard-classifier", stage="Production")
```

**Full documentation:** [Model Registry Guide](docs/model_registry_guide.md)

## Project Status

🔄 **In Progress** — Phase III: Model Deployment - API Deployment

## Milestones
- [ ] 3.1: FastAPI Project Setup
- [ ] 3.2: Model Loading & Production Endpoint
- [ ] 3.3: Input Validation & Error Handling
- [ ] 3.4: Docker Containerization
- [ ] 3.5: API Documentation & Testing

---


## Installation

### Prerequisites
- Python 3.10+
- pip
- virtualenv (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/muhasin-code/churnguard-ai.git
cd churnguard-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import mlflow, fastapi, xgboost; print('✅ Installation successful')"
```

### Dependency Management

**Files:**
- `requirements.txt` - Main dependency file (install this)
- `requirements-frozen.txt` - Exact versions (for reproducibility)

**Update dependencies:**
```bash
# Add new package
echo "new-package==1.0.0" >> requirements.txt
pip install -r requirements.txt

# Lock versions
pip freeze > requirements-frozen.txt
```

**Reproduce exact environment:**
```bash
pip install -r requirements-frozen.txt
```

## Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 1.29+

### Quick Start

**Start all services:**
```bash
./scripts/docker-start.sh
```

**Access:**
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

**Stop services:**
```bash
./scripts/docker-stop.sh
```

### Manual Docker Commands

**Build images:**
```bash
docker-compose build
```

**Start services:**
```bash
docker-compose up -d
```

**View logs:**
```bash
# All services
docker-compose logs -f

# API only
docker-compose logs -f api

# MLflow only
docker-compose logs -f mlflow
```

**Stop services:**
```bash
docker-compose down
```

**Clean up everything:**
```bash
docker-compose down -v  # Remove volumes too
```

### Container Architecture
┌─────────────────────────────────────┐
│     docker-compose.yml              │
├─────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐ │
│  │   MLflow     │  │     API     │ │
│  │  Port: 5000  │  │  Port: 8000 │ │
│  └──────┬───────┘  └──────┬──────┘ │
│         │                 │         │
│         └────────┬────────┘         │
│         churnguard-network          │
└─────────────────────────────────────┘

### Troubleshooting

**Container won't start:**
```bash
# Check logs
docker-compose logs api

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

**Model not found:**
```bash
# Verify model file exists
docker-compose exec api ls -lh /app/models/

# Check MLflow connection
docker-compose exec api curl http://mlflow:5000/health
```

**API not responding:**
```bash
# Check health
curl http://localhost:8000/health/live

# Enter container
docker-compose exec api bash

# Test internally
curl http://localhost:8000/health
```

## Testing

### Running Tests

**All tests:**
```bash
./scripts/run_tests.sh
```

**Fast tests only (skip performance tests):**
```bash
./scripts/run_tests_fast.sh
```

**Specific test file:**
```bash
pytest tests/api/routers/test_health.py -v
```

**With coverage report:**
```bash
pytest --cov=src/api --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Test Coverage

Current coverage: **91%**

Coverage breakdown:
- `src/api/models.py`: 95%
- `src/api/routers/`: 92-94%
- `src/api/services/`: 90%
- `src/api/config.py`: 88%

### Test Categories

Tests are marked with categories:
- `unit`: Fast, isolated tests
- `integration`: Tests requiring services
- `slow`: Performance tests

**Run specific category:**
```bash
pytest -m unit          # Unit tests only
pytest -m "not slow"    # Skip slow tests
pytest -m integration   # Integration tests only
```

### API Testing with Postman

**Import collection:**
1. Open Postman
2. Import `postman/ChurnGuard_API.postman_collection.json`
3. Set `base_url` variable to `http://localhost:8000`

**Available requests:**
- Health checks
- Single/batch predictions
- Validation test cases

See `postman/README.md` for details.

## Quick Start

```bash
# 1. Generate data
python scripts/generate_synthetic_data_v3.1.py

# 2. Run feature engineering
python scripts/engineer_features.py

# 3. Train models
python src/models/train_models.py

# 4. Evaluate champion
python scripts/save_champion_from_mlflow.py
python scripts/evaluate_champion.py
```

---

## Author
Muhammed Muhasin K
LinkedIn — https://www.linkedin.com/in/muhasin-code

## License
MIT License