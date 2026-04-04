# ChurnGuard AI

> Production-grade customer churn prediction system with full MLOps pipeline

## Problem Statement
Customer churn costs SaaS companies millions annually. ChurnGuard AI predicts which customers are likely to churn 30 days in advance, enabling proactive retention campaigns.

## Business Impact
- **Early detection:** Identify at-risk customers 30 days before churn
- **Cost savings:** Reduce churn rate by 15% (estimated $X saved annually)
- **Efficiency:** Automated pipeline reduces manual analysis time 90%

## System Architecture
[Add diagram later]

## Tech Stack
- **ML Framework:** scikit-learn, XGBoost, LightGBM
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Orchestration:** Prefect
- **API:** FastAPI
- **Deployment:** Docker, AWS
- **Monitoring:** Evidently, Prometheus, Grafana

## Data Generation

**Version History:**
- `v1` (buggy): Had negative CallMinutes/DataUsage, dtype inconsistencies
- `v2` (current): Fixed np.clip(), proper dtype enforcement

**Known issues addressed:**
1. Negative usage values -> clipped to 0
2. InternetService nulls -> represent "No Service" customers
3. dtype consistency -> enforced at generation time

## Project Status
🚧 **In Development** - Phase I: Data Foundation

## Milestones
- [x] Project setup
- [ ] Data pipeline
- [ ] Model development
- [ ] MLOps infrastructure
- [ ] API deployment
- [ ] Production Monitoring

## Quick Start
Coming soon...

## Author
Muhammed Muhasin K
LinkedIn - https://www.linkedin.com/in/muhasin-code

## License
MIT License