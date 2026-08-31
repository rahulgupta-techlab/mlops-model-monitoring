"""Data-drift metrics: Population Stability Index (PSI) and Kolmogorov-Smirnov test.

Both functions compare a reference (baseline / training) distribution of a single
numeric feature against a production (current / incoming) distribution and return
a scalar drift signal.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

# Default alert thresholds, used by the API layer.
PSI_ALERT_THRESHOLD = 0.2
KS_PVALUE_ALERT_THRESHOLD = 0.05


def compute_psi(reference: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:
    """Compute the Population Stability Index between two 1-D numeric arrays.

    PSI buckets the reference distribution into `buckets` equal-frequency bins
    (using reference quantiles as bin edges), then compares the proportion of
    reference vs. production observations that fall into each bin:

        PSI = sum( (prod_pct - ref_pct) * ln(prod_pct / ref_pct) )

    Rules of thumb for interpreting PSI:
        < 0.1  -> no significant shift
        0.1-0.2 -> moderate shift, worth investigating
        > 0.2  -> significant shift

    Returns 0.0 for two empty arrays, and a large finite value if a bucket is
    populated on one side only (rather than raising on log(0)).
    """
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)

    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]

    if reference.size == 0 or production.size == 0:
        return 0.0

    # Build bin edges from reference quantiles so each reference bucket has
    # (approximately) equal frequency. Guard against degenerate/constant data.
    quantiles = np.linspace(0, 100, buckets + 1)
    edges = np.percentile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        # Reference feature is constant; fall back to a single bucket comparison.
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    prod_counts, _ = np.histogram(production, bins=edges)

    ref_pct = ref_counts / reference.size
    prod_pct = prod_counts / production.size

    # Avoid division by zero / log(0) by flooring proportions at a small epsilon.
    epsilon = 1e-4
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    prod_pct = np.where(prod_pct == 0, epsilon, prod_pct)

    psi_value = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi_value)


def compute_ks(reference: np.ndarray, production: np.ndarray) -> tuple[float, float]:
    """Run a two-sample Kolmogorov-Smirnov test between two 1-D numeric arrays.

    Returns (statistic, p_value). A low p-value (e.g. < 0.05) indicates the two
    samples are unlikely to be drawn from the same distribution, i.e. drift.
    """
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)

    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]

    if reference.size == 0 or production.size == 0:
        return 0.0, 1.0

    result = ks_2samp(reference, production)
    return float(result.statistic), float(result.pvalue)


def evaluate_feature_drift(
    reference: np.ndarray,
    production: np.ndarray,
    psi_threshold: float = PSI_ALERT_THRESHOLD,
    ks_pvalue_threshold: float = KS_PVALUE_ALERT_THRESHOLD,
) -> dict:
    """Compute PSI and KS-test results for one feature and flag drift.

    A feature is flagged as drifted if PSI exceeds `psi_threshold` OR the KS
    p-value falls below `ks_pvalue_threshold`.
    """
    psi_value = compute_psi(reference, production)
    ks_statistic, ks_pvalue = compute_ks(reference, production)

    drifted = bool(psi_value > psi_threshold or ks_pvalue < ks_pvalue_threshold)

    return {
        "psi": round(psi_value, 6),
        "ks_statistic": round(ks_statistic, 6),
        "ks_pvalue": round(ks_pvalue, 6),
        "drifted": drifted,
    }
