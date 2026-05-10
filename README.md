# E-commerce Lifecycle Engine

An ML-powered e-commerce churn prediction system using:
- Google BigQuery
- Scikit-learn Pipelines
- XGBoost
- Weights & Biases
- FastAPI

## Project Structure

- `src/` → Core ML training and feature engineering
- `api/` → FastAPI deployment layer
- `data/` → Local sample datasets
- `notebooks/` → EDA and experimentation

## Installation

```bash
pip install -r requirements.txt
```

## Run API

```bash
uvicorn api.main:app --reload
```
