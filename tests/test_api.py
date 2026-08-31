import pandas as pd
from fastapi.testclient import TestClient

from src.main import BASE_DIR, app

client = TestClient(app)


def _load_rows(csv_name: str) -> list[dict]:
    df = pd.read_csv(BASE_DIR / "data" / csv_name)
    return df.to_dict(orient="records")


def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200


def test_check_drift_with_reference_like_batch_returns_200_and_no_alert():
    rows = _load_rows("reference_data.csv")

    response = client.post("/check-drift", json={"rows": rows})

    assert response.status_code == 200
    body = response.json()
    assert "alert" in body
    assert "features" in body
    assert body["n_reference_rows"] > 0
    assert body["n_production_rows"] == len(rows)
    for feature_name, result in body["features"].items():
        assert set(result.keys()) == {"psi", "ks_statistic", "ks_pvalue", "drifted"}


def test_check_drift_with_drifted_batch_triggers_alert():
    rows = _load_rows("drifted_data.csv")

    response = client.post("/check-drift", json={"rows": rows})

    assert response.status_code == 200
    body = response.json()
    assert body["alert"] is True
    assert any(f["drifted"] for f in body["features"].values())


def test_check_drift_response_shape_matches_reference_columns():
    rows = _load_rows("reference_data.csv")

    response = client.post("/check-drift", json={"rows": rows})

    body = response.json()
    reference_df = pd.read_csv(BASE_DIR / "data" / "reference_data.csv")
    numeric_cols = set(reference_df.select_dtypes(include="number").columns)
    assert set(body["features"].keys()) == numeric_cols
