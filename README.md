# Credit Scoring API

A machine learning service that predicts credit risk (good / bad) for loan applicants. The project covers the full path from exploratory data analysis and model experiments to a containerized REST API deployed on Railway.

**Live API:** https://credit-scoring-api-production-17f6.up.railway.app
**Swagger UI:** https://credit-scoring-api-production-17f6.up.railway.app/docs

## Overview

The project builds a binary credit scoring model on the German Credit Data (Statlog) dataset, with a focus on **recall for the bad class**. In lending, missing a risky applicant is usually more expensive than rejecting a reliable one, so detecting bad applicants was the main selection criterion.

Several approaches were compared, and a tuned CatBoost model was chosen as the final one. It is served through a FastAPI service, packaged with Docker and deployed on Railway.

## Pipeline

```
EDA → Baseline models → CatBoost → Class weights → Hyperparameter tuning
   → SHAP interpretation → Feature engineering experiments → Final CatBoost model
   → FastAPI → Docker → Railway
```

## Dataset

German Credit Data (Statlog):

- 1,000 applicants, 19 features (categorical and numerical)
- binary target `creditability`: 1 = good, 0 = bad
- roughly 700 good / 300 bad

The `foreign_worker` column was dropped during EDA as a protected attribute (a fairness decision, not a data-leakage one).

## Model Performance

Models were compared primarily by recall for the bad class, with precision and ROC-AUC as secondary metrics.

| Model | Recall (bad) | Precision (bad) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline, balanced) | 0.77 | 0.54 | 0.8294 |
| CatBoost (default) | 0.43 | 0.67 | 0.8324 |
| CatBoost (balanced class weights) | 0.67 | 0.62 | 0.8285 |
| **CatBoost (tuned, final)** | **0.67** | **0.62** | **0.8335** |

Hyperparameters were tuned with `RandomizedSearchCV` (5-fold cross-validation, 30 iterations).

## Interpretation

SHAP analysis shows the strongest predictors are:

- `account_balance`
- `value_savings_stocks`
- `duration_of_credit_monthly`

This matches domain intuition: an applicant's financial position and the loan terms are closely tied to default risk.

Feature engineering experiments (removing weak features by SHAP, adding manual interactions) did not improve on the tuned CatBoost, which is expected since tree models find non-linear interactions on their own.

## API

### `POST /predict`

Accepts a loan application and returns:

- `creditability`: predicted class (1 = good, 0 = bad), using a 0.5 probability threshold
- `probability_good`: model probability of the good class
- `risk_level`: `low` (probability ≥ 0.7), `medium` (0.4 to 0.7) or `high` (< 0.4)

The service also computes the derived feature `credit_amount_per_month` (credit amount divided by duration), the same way as during training. Input is validated with Pydantic; invalid or missing fields return `422`.

**Example request:**

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "account_balance": 1,
    "duration_of_credit_monthly": 18,
    "payment_status_of_previous_credit": 4,
    "purpose": 2,
    "credit_amount": 1049,
    "value_savings_stocks": 1,
    "length_of_current_employment": 2,
    "instalment_per_cent": 4,
    "sex_marital_status": 2,
    "guarantors": 1,
    "duration_in_current_address": 4,
    "most_valuable_available_asset": 2,
    "age_years": 21,
    "concurrent_credits": 3,
    "type_of_apartment": 1,
    "no_of_credits_at_this_bank": 1,
    "occupation": 3,
    "no_of_dependents": 1,
    "telephone": 1
  }'
```

**Example response:**

```json
{
  "creditability": 0,
  "probability_good": 0.2242,
  "risk_level": "high"
}
```

### `GET /health`

Health check; returns `{"status": "ok"}`.

Interactive documentation is available at `/docs` (Swagger UI), including on the [live deployment](https://credit-scoring-api-production-17f6.up.railway.app/docs).

## Project Structure

```text
credit-scoring-api/
├── app/
│   ├── main.py             # FastAPI app and endpoints
│   ├── schemas.py          # Pydantic request/response models
│   ├── model_loader.py     # CatBoost model loading and caching
│   └── config.py           # Paths, thresholds, feature order
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned dataset
├── models/                 # Trained CatBoost model (.cbm)
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory data analysis
│   └── 02_modeling.ipynb   # Baselines, CatBoost, tuning, SHAP
├── tests/
│   └── test_api.py         # API tests
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── requirements.txt
```

## Running Locally

```bash
pip install -r requirements.txt
python -m app.main
```

The API will be available at `http://127.0.0.1:8000`, Swagger UI at `http://127.0.0.1:8000/docs`.

## Running with Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest tests/ -v
```

The tests cover the health check, valid requests, missing and out-of-range fields (validation errors), and consistency between `risk_level` and the probability thresholds.

## Tech Stack

Python, pandas, scikit-learn, CatBoost, SHAP, FastAPI, Pydantic, pytest, Docker, Railway