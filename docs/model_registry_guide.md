# Model Registry Guide - ChurnGuard AI

**Date:** 2026-04-18  
**Model:** churnguard-classifier  
**Current Production Version:** v1  
**ROC-AUC:** 0.8077

---

## Overview

This guide documents the MLflow Model Registry setup for ChurnGuard AI, including registration workflow, version management, and deployment procedures.

---

## Registered Models

### churnguard-classifier

**Purpose:** Production churn prediction model for telecom customers

**Model Type:** XGBoost Classifier  
**Training Data:** v3.1 (50,000 synthetic telecom customers)  
**Features:** 24 engineered features  
**Target:** Binary classification (Churn: Yes/No)

---

## Version History

| Version | Stage | Data | ROC-AUC | Date | Notes |
|---------|-------|------|---------|------|-------|
| 1 | Production | v3.1 | 0.8077 | 2026-04-18 | Initial production deployment |

### Version 1 Details

**Training Run:** `xgb-conservative-20260418-200510`  
**Run ID:** `20e971f66ba64291a8a6f6763418f5ed`

**Performance Metrics:**
- ROC-AUC: 0.8077
- Accuracy: 73.66%
- Precision: 83.60%
- Recall: 74.14%
- F1 Score: 0.7859
- PR-AUC: 0.8786

**Hyperparameters:**
```python
n_estimators=200
learning_rate=0.05
max_depth=4
min_child_weight=3
gamma=0.1
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=1.0
scale_pos_weight=0.539
```

**Tags:**
- `data_version`: v3.1
- `champion`: true
- `model_type`: XGBoost

---

## Deployment Workflow

### 1. Model Registration

```python
from src.models.registry_utils import ModelRegistry

registry = ModelRegistry()

# Register from MLflow run
version = registry.register_champion(
    run_id="<mlflow_run_id>",
    model_name="churnguard-classifier",
    data_version="v3.1",
    performance_metrics={
        "roc_auc": 0.8077,
        "accuracy": 0.7366,
        ...
    }
)
```

**Automated script:**
```bash
python scripts/register_champion.py
```

---

### 2. Staging Validation

**Purpose:** Test model in pre-production environment

```bash
python scripts/validate_staging_model.py
```

**Validation checks:**
- ✅ Model signature (predict, predict_proba methods)
- ✅ Prediction sanity (no NaN, valid probabilities)
- ✅ Performance criteria (ROC-AUC ≥ 0.80, Recall ≥ 0.70)

---

### 3. Production Promotion

**After validation passes:**

```bash
python scripts/promote_to_production.py
```

**Effect:**
- Moves model from Staging → Production
- Archives previous production version automatically
- Updates registry metadata

---

## Loading Models for Inference

### Load Production Model

```python
from src.models.registry_utils import ModelRegistry

registry = ModelRegistry()

# Load latest production model
model = registry.load_model("churnguard-classifier", stage="Production")

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

### Load Specific Version

```python
# Load version 1 specifically
model = registry.load_model("churnguard-classifier", version=1)
```

---

## Model Comparison

### Compare Multiple Versions

```python
registry = ModelRegistry()

comparison = registry.compare_versions(
    model_name="churnguard-classifier",
    versions=[1, 2, 3],
    metrics=["test_roc_auc", "test_accuracy", "test_recall"]
)

print(comparison)
```

**Output:**
```
 version      stage  test_roc_auc  test_accuracy  test_recall
       1 Production        0.8077         0.7366       0.7414
       2    Staging        0.8095         0.7401       0.7523
       3       None        0.7921         0.7201       0.7012
```

---

## Rollback Procedure

**If production model fails:**

```python
registry = ModelRegistry()

# Archive failing production model
registry.archive_version("churnguard-classifier", version=2)

# Promote previous version back to production
registry.promote_to_production("churnguard-classifier", version=1)
```

---

## Maintenance

### Archive Old Versions

```python
# Archive versions no longer needed
registry.archive_version("churnguard-classifier", version=3)
```

### Update Metadata

```python
# Add tags
registry.add_tags(
    "churnguard-classifier",
    version=1,
    tags={"validated_by": "muhasin", "deployment_date": "2026-04-18"}
)

# Update description
registry.update_description(
    "churnguard-classifier",
    version=1,
    description="Production model - validated and approved"
)
```

---

## Best Practices

### Naming Conventions

**Registered Model Names:**
- Format: `{project}-{model-type}`
- Example: `churnguard-classifier`
- Use lowercase with hyphens

**Version Tags:**
- `data_version`: Training data version (e.g., "v3.1")
- `champion`: "true" for best model
- `approved_by`: Name of approver
- `deployment_date`: Date promoted to production

### Stage Transitions

**Recommended workflow:**
```
None (new) → Staging (testing) → Production (live) → Archived (deprecated)
```

**Never skip Staging!** Always validate before production.

### Validation Criteria

**Minimum thresholds for production:**
- ROC-AUC ≥ 0.80
- Recall ≥ 0.70
- No NaN predictions
- Probabilities in [0, 1] range

---

## MLflow UI Access

**Local:** http://localhost:5000  
**Navigate to:** Models tab → churnguard-classifier

**What you can see:**
- All registered versions
- Current stages (Staging, Production, Archived)
- Performance metrics
- Model lineage (source run)
- Deployment history

---

## Troubleshooting

### Model Not Found

**Error:** `RESOURCE_DOES_NOT_EXIST: Registered model 'churnguard-classifier' not found`

**Solution:**
1. Check model name spelling
2. Verify model is registered: `registry.list_registered_models()`
3. Register if missing: `python scripts/register_champion.py`

### No Staging Version

**Error:** `No model in Staging for 'churnguard-classifier'`

**Solution:**
1. Check current versions: `registry.get_model_versions("churnguard-classifier")`
2. Promote to staging: `registry.promote_to_staging(model_name, version)`

### Validation Fails

**Check:**
1. Test data exists: `data/processed/features_v1_test.csv`
2. Model meets criteria (ROC-AUC ≥ 0.80, Recall ≥ 0.70)
3. Predictions are valid (no NaN, probabilities in [0, 1])

---

## Next Steps

- [ ] Deploy model to FastAPI endpoint (Milestone 3.1)
- [ ] Set up monitoring dashboard (Milestone 4.1)
- [ ] Configure automated retraining (Milestone 4.2)
- [ ] Document deployment architecture

---

## References

- [MLflow Model Registry Docs](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/mlflow.html)
- Project repo: https://github.com/muhasin-code/churnguard-ai