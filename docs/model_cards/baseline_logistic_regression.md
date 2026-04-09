# Model Card: Baseline Logistic Regression

**Model Type:** Binary Classification (Churn Prediction)
**Algorithm:** Logistic Regression with L2 Regularization
**Training Date:** 2026-04-09
**Model Version:** v1.0 (baseline)
**Status:** ✅ Baseline Established

---

## Model Overview

### Purpose

Establish minimum acceptable performance for customer churn prediction. This baseline serves as the benchmark against which all advanced models will be compared.

### Model Details

* **Framework:** scikit-learn 1.3.0
* **Algorithm:** Logistic Regression
* **Regularization:** L2 (Ridge)
* **Class Balancing:** Balanced class weights to handle class imbalance

### Training Configuration

```python
LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

---

## Data

### Training Data

* **Source:** `data/processed/features_v1_train.csv`
* **Samples:** 40,000 customers
* **Features:** 24 (from 25 columns including target)
* **Churn Rate:** 64.96% (25,983 churned, 14,017 retained)
* **Data Version:** v1.0

### Test Data

* **Source:** `data/processed/features_v1_test.csv`
* **Samples:** 10,000 customers
* **Churn Rate:** 64.96% (6,496 churned, 3,504 retained)
* **Split Strategy:** Stratified random split (80/20)

### Feature List

All 24 features were used (no feature selection applied).

---

## Performance

### Metrics Summary

| Metric        | Train  | Test       |
| ------------- | ------ | ---------- |
| **Accuracy**  | 0.7315 | 0.7271     |
| **Precision** | -      | 0.8292     |
| **Recall**    | -      | 0.7303     |
| **F1 Score**  | -      | 0.7766     |
| **ROC-AUC**   | -      | **0.8003** |
| **PR-AUC**    | -      | 0.8725     |

### Classification Report (Test)

```
              precision    recall  f1-score   support

    No Churn       0.59      0.72      0.65      3504
       Churn       0.83      0.73      0.78      6496

    accuracy                           0.73     10000
   macro avg       0.71      0.73      0.71     10000
weighted avg       0.75      0.73      0.73     10000
```

### Performance Interpretation

**Strengths:**

* ROC-AUC of 0.8003 indicates strong predictive power
* Good balance between precision (0.8292) and recall (0.7303)
* No overfitting (train ≈ test performance)

**Limitations:**

* Misses ~27% of churners
* Moderate false positives for non-churners
* Linear model limits interaction capture

---

## Confusion Matrix (Derived)

```
                  Predicted
                No      Yes
Actual  No    2523    981
        Yes   1753    4743
```

### Business Impact

* **False Negatives (~1,753):** Missed churners → revenue loss risk
* **False Positives (~981):** Unnecessary retention actions → cost overhead

---

## Feature Importance

### Top 10 Most Predictive Features

| Rank | Feature               | Coefficient | Importance | Direction                          |
| ---- | --------------------- | ----------- | ---------- | ---------------------------------- |
| 1    | ContractType_Two Year | -1.9086     | 1.9086     | Strongly reduces churn             |
| 2    | ContractType_One Year | -1.5017     | 1.5017     | Reduces churn                      |
| 3    | RecentSupportTickets  | +1.4239     | 1.4239     | Strong churn indicator             |
| 4    | Tenure                | -0.6787     | 0.6787     | Longer tenure reduces churn        |
| 5    | FinancialStress       | +0.3551     | 0.3551     | Increases churn                    |
| 6    | MonthlyCharges        | +0.3439     | 0.3439     | Higher cost increases churn        |
| 7    | Engagement            | -0.3141     | 0.3141     | Higher engagement reduces churn    |
| 8    | Complaints            | +0.2852     | 0.2852     | Complaints increase churn          |
| 9    | IsHighRisk            | -0.0916     | 0.0916     | Weak effect (unexpected direction) |
| 10   | TotalCharges          | +0.0837     | 0.0837     | Slight increase in churn           |

### Insights

* Long-term contracts (1-year, 2-year) are the strongest retention factors
* Customer support interactions (tickets, complaints) are major churn signals
* Engagement reduces churn significantly
* Financial and pricing factors remain important drivers

---

## Evaluation Criteria

### Business Requirements

* ROC-AUC ≥ 0.75 → ✅ 0.8003
* Recall ≥ 0.70 → ✅ 0.7303
* Precision ≥ 0.60 → ✅ 0.8292

### Technical Requirements

* No overfitting → ✅
* Interpretable → ✅
* Fast inference → ✅
* Reproducible → ✅

---

## Comparison to Naive Baselines

| Baseline Strategy      | Accuracy   | ROC-AUC    |
| ---------------------- | ---------- | ---------- |
| Always predict "Churn" | ~0.65      | 0.50       |
| Random prediction      | 0.50       | 0.50       |
| Logistic Regression    | **0.7271** | **0.8003** |

---

## Limitations & Future Work

### Known Limitations

1. **Linear Model Constraints**
2. **No Feature Interactions**
3. **Class Imbalance Handling Not Optimal**
4. **No Temporal Awareness**

### Recommendations

**Priority 1:**

* Train Random Forest
* Train XGBoost
* Compare models in MLflow

**Priority 2:**

* Add interaction features
* Polynomial features
* Threshold optimization

**Priority 3:**

* Hyperparameter tuning (Optuna)
* Ensemble methods
* Probability calibration

---

## Deployment Considerations

### Model Artifacts

* **Model Path:** `models/baseline_logistic_regression.pkl`
* **MLflow Run:** `logistic-baseline-20260409-193544`
* **Dependencies:** scikit-learn==1.3.0

### Inference Performance

* Latency: <5ms
* Throughput: >1000 predictions/sec
* Memory: <10MB

### Production Readiness: 🟡 PARTIAL

**Ready:**

* Stable baseline
* Meets performance thresholds
* Interpretable

**Not Ready:**

* Not best-performing model
* No monitoring or A/B testing

---

## Ethical Considerations

### Fairness

* Potential bias via demographic features
* Requires subgroup performance validation

### Transparency

* Fully interpretable model
* Coefficients explainable

### Privacy

* No PII used
* Uses behavioral and aggregated data

---

## Reproducibility

```bash
python src/models/train_baseline.py
```

* Random seed fixed
* Data versioned (DVC)
* Code versioned (Git)
* Deterministic output

---

## Changelog

| Version | Date       | Changes                                   | Author  |
| ------- | ---------- | ----------------------------------------- | ------- |
| v1.0    | 2026-04-09 | Updated metrics, features, and MLflow run | Muhasin |

---

## Approval

**Status:** ✅ Baseline Established
**Approved For:** Benchmarking advanced models
**Next Review:** After advanced model experimentation (RF, XGBoost)
