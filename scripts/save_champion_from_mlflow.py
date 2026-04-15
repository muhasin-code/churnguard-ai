"""
Script to save the Champion model from MLflow to locally.

Run this before scripts/evaluate_champion.py
"""

import mlflow
import joblib
from pathlib import Path

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get champion run ID
# Replace with your actual run ID from MLflow UI
champion_run_id = "YOUR_RUN_ID_GOES_HERE" # e.g., "09852867855742aca3055b4a2c3f3c9e"

print(f"Loading model from run: {champion_run_id}")

# Load model from MLflow
model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")

# Save loaclly
output_path = Path("models/xgboost_conservative.pkl")
output_path.parent.mkdir(exist_ok=True)

joblib.dump(model, output_path)
print(f"Saved champion model to {output_path}")