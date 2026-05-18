# ChurnGuard AI - Postman Collection

## Import Instructions

1. Open Postman
2. Click "Import" button
3. Select `ChurnGuard_API.postman_collection.json`
4. Collection will appear in your workspace

## Environment Variables

The collection uses one variable:
- `base_url`: API base URL (default: `http://localhost:8000`)

### Change Base URL

1. Click on collection name
2. Go to "Variables" tab
3. Update `base_url` value

## Available Requests

### Health Checks
- **Basic Health Check** - GET /health/
- **Readiness Check** - GET /health/ready
- **Liveness Check** - GET /health/live

### Predictions
- **Single Prediction - Low Risk** - Example low-risk customer
- **Single Prediction - High Risk** - Example high-risk customer
- **Batch Prediction** - Predict multiple customers
- **Model Info** - Get model metadata

### Validation Tests
- **Invalid Age** - Tests age validation (< 18)
- **Invalid Total Charges** - Tests total_charges consistency

## Running the Collection

### Manual Testing
1. Start the API (`./scripts/docker-start.sh` or `./scripts/run_api.sh`)
2. Select a request
3. Click "Send"
4. View response

### Automated Testing (Collection Runner)
1. Click collection name
2. Click "Run" button
3. Select requests to run
4. Click "Run ChurnGuard AI"
5. View test results

## Expected Responses

### Successful Prediction
```json
{
  "customer_id": "...",
  "churn_prediction": "Yes" or "No",
  "churn_probability": 0.0 - 1.0,
  "risk_level": "Low|Medium|High|Critical",
  "confidence": 0.0 - 1.0,
  "model_version": "v1",
  "prediction_date": "2026-05-15T..."
}
```

### Validation Error
```json
{
  "error": "ValidationError",
  "message": "Request validation failed",
  "errors": [...]
}
```