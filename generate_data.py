"""Generate the sample reference and drifted datasets used by this demo.

Run once with `python generate_data.py` to (re)create data/reference_data.csv and
data/drifted_data.csv. A fixed numpy seed keeps the output reproducible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 500

DATA_DIR = Path(__file__).resolve().parent / "data"


def generate_reference(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_age_days": rng.normal(loc=400, scale=80, size=n).clip(min=0),
            "monthly_spend": rng.normal(loc=120, scale=25, size=n).clip(min=0),
            "num_transactions": rng.poisson(lam=15, size=n),
            "credit_score": rng.normal(loc=680, scale=45, size=n).clip(300, 850),
        }
    )


def generate_drifted(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # Deliberately shift mean/std/lambda so PSI and KS pick up a clear signal:
    # older accounts, much higher spend and variance, more transactions, lower
    # (and more spread out) credit scores.
    return pd.DataFrame(
        {
            "account_age_days": rng.normal(loc=520, scale=110, size=n).clip(min=0),
            "monthly_spend": rng.normal(loc=210, scale=60, size=n).clip(min=0),
            "num_transactions": rng.poisson(lam=27, size=n),
            "credit_score": rng.normal(loc=600, scale=70, size=n).clip(300, 850),
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ref_rng = np.random.default_rng(SEED)
    reference_df = generate_reference(ref_rng, N_ROWS)
    reference_df.to_csv(DATA_DIR / "reference_data.csv", index=False)

    drift_rng = np.random.default_rng(SEED + 1)
    drifted_df = generate_drifted(drift_rng, N_ROWS)
    drifted_df.to_csv(DATA_DIR / "drifted_data.csv", index=False)

    print(f"Wrote {len(reference_df)} rows to {DATA_DIR / 'reference_data.csv'}")
    print(f"Wrote {len(drifted_df)} rows to {DATA_DIR / 'drifted_data.csv'}")


if __name__ == "__main__":
    main()
