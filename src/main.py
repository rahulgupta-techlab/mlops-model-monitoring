"""FastAPI service exposing POST /check-drift.

Loads a reference dataset (data/reference_data.csv) at startup and compares it,
feature by feature, against a batch of production rows sent in the request body.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.drift_metrics import evaluate_feature_drift

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DATA_PATH = BASE_DIR / "data" / "reference_data.csv"

app = FastAPI(
    title="MLOps Model Monitoring",
    description="Data-drift detection service comparing production batches to a reference dataset.",
    version="0.1.0",
)


def _load_reference_data() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DATA_PATH)


REFERENCE_DF = _load_reference_data()


class DriftCheckRequest(BaseModel):
    """A batch of production rows to compare against the reference dataset.

    `rows` is a list of JSON objects, one per production observation, e.g.:
        {"rows": [{"feature_a": 1.2, "feature_b": 5.0}, ...]}
    Only numeric columns that also exist in the reference dataset are scored.
    """

    rows: list[dict[str, Any]]


class FeatureDriftResult(BaseModel):
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


class DriftCheckResponse(BaseModel):
    alert: bool
    features: dict[str, FeatureDriftResult]
    n_reference_rows: int
    n_production_rows: int


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "mlops-model-monitoring"}


@app.post("/check-drift", response_model=DriftCheckResponse)
def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    production_df = pd.DataFrame(request.rows)

    numeric_reference_cols = REFERENCE_DF.select_dtypes(include="number").columns
    shared_cols = [c for c in numeric_reference_cols if c in production_df.columns]

    features: dict[str, FeatureDriftResult] = {}
    for col in shared_cols:
        result = evaluate_feature_drift(
            REFERENCE_DF[col].to_numpy(),
            pd.to_numeric(production_df[col], errors="coerce").to_numpy(),
        )
        features[col] = FeatureDriftResult(**result)

    alert = any(f.drifted for f in features.values())

    return DriftCheckResponse(
        alert=alert,
        features=features,
        n_reference_rows=len(REFERENCE_DF),
        n_production_rows=len(production_df),
    )
