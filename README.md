# MLOps Model Monitoring

A small, self-contained data-drift detection service for monitoring machine learning
models in production. It compares a reference (training/baseline) dataset to incoming
production batches, feature by feature, using two standard statistical drift metrics -
the Population Stability Index (PSI) and a two-sample Kolmogorov-Smirnov (KS) test -
and raises an alert when any feature has drifted beyond a threshold.

Built to demonstrate the core model-monitoring pattern end-to-end: drift math, a REST
API for scoring incoming batches, and reproducible sample data (one dataset that
matches the reference distribution, one deliberately shifted) to exercise both the
"no drift" and "drift detected" paths.

## Stack

- Python 3.11
- `numpy` / `scipy` for the PSI and KS-test drift metrics
- `pandas` for dataset handling
- FastAPI for the `/check-drift` endpoint

## Project layout

```
src/
  drift_metrics.py   # PSI + KS-test functions, feature-level drift evaluation
  main.py             # FastAPI app exposing POST /check-drift
data/
  reference_data.csv  # baseline dataset (numeric features)
  drifted_data.csv     # sample batch with a deliberate distribution shift
generate_data.py       # regenerates both CSVs from a fixed numpy seed
tests/
  test_drift_metrics.py  # unit tests for PSI / KS-test / drift evaluation
  test_api.py              # FastAPI endpoint tests (TestClient)
```

## Running it

```bash
pip install -r requirements.txt
python generate_data.py              # (optional) regenerate data/*.csv
uvicorn src.main:app --reload        # serves POST /check-drift
```

A minimal request body looks like:

```json
{
  "rows": [
    {"account_age_days": 410, "monthly_spend": 118, "num_transactions": 14, "credit_score": 690}
  ]
}
```

```bash
curl -X POST http://localhost:8000/check-drift -H "Content-Type: application/json" \
  -d '{"rows": [{"account_age_days": 410, "monthly_spend": 118, "num_transactions": 14, "credit_score": 690}]}'
```

The response reports PSI, the KS statistic, and the KS p-value per feature, plus an
overall `alert` flag (true if any feature's PSI > 0.2 or KS p-value < 0.05):

```json
{
  "alert": false,
  "features": {
    "account_age_days": {"psi": 0.01, "ks_statistic": 0.03, "ks_pvalue": 0.87, "drifted": false}
  },
  "n_reference_rows": 500,
  "n_production_rows": 1
}
```

## Tests

```bash
pytest tests/
```

Tests cover the drift metrics directly (identical distributions score low, clearly
shifted ones score high and trigger the alert) as well as the `/check-drift` endpoint
(200 response, correct response shape, and that the shipped drifted dataset actually
trips the alert).

## Notes

This is a portfolio/demo project - the datasets are synthetic (generated with numpy,
fixed seed for reproducibility) and the drift thresholds are simple defaults, but the
PSI and KS-test implementations, and the API contract around them, are real and meant
to reflect how a lightweight monitoring service would be structured in practice.
