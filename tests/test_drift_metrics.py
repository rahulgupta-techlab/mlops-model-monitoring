import numpy as np

from src.drift_metrics import compute_ks, compute_psi, evaluate_feature_drift


def test_psi_identical_distributions_is_low():
    rng = np.random.default_rng(1)
    reference = rng.normal(loc=100, scale=10, size=1000)
    production = rng.normal(loc=100, scale=10, size=1000)

    psi_value = compute_psi(reference, production)

    assert psi_value < 0.1


def test_psi_shifted_distribution_is_high():
    rng = np.random.default_rng(2)
    reference = rng.normal(loc=100, scale=10, size=1000)
    production = rng.normal(loc=160, scale=30, size=1000)

    psi_value = compute_psi(reference, production)

    assert psi_value > 0.2


def test_ks_identical_distributions_has_high_pvalue():
    rng = np.random.default_rng(3)
    reference = rng.normal(loc=50, scale=5, size=1000)
    production = rng.normal(loc=50, scale=5, size=1000)

    statistic, pvalue = compute_ks(reference, production)

    assert pvalue >= 0.05
    assert statistic < 0.1


def test_ks_shifted_distribution_has_low_pvalue():
    rng = np.random.default_rng(4)
    reference = rng.normal(loc=50, scale=5, size=1000)
    production = rng.normal(loc=90, scale=15, size=1000)

    statistic, pvalue = compute_ks(reference, production)

    assert pvalue < 0.05
    assert statistic > 0.3


def test_evaluate_feature_drift_flags_no_drift_for_identical_distributions():
    rng = np.random.default_rng(5)
    reference = rng.normal(loc=0, scale=1, size=500)
    production = rng.normal(loc=0, scale=1, size=500)

    result = evaluate_feature_drift(reference, production)

    assert result["drifted"] is False
    assert result["psi"] < 0.2
    assert result["ks_pvalue"] >= 0.05


def test_evaluate_feature_drift_flags_drift_for_shifted_distributions():
    rng = np.random.default_rng(6)
    reference = rng.normal(loc=0, scale=1, size=500)
    production = rng.normal(loc=5, scale=3, size=500)

    result = evaluate_feature_drift(reference, production)

    assert result["drifted"] is True


def test_psi_handles_empty_arrays():
    assert compute_psi(np.array([]), np.array([])) == 0.0


def test_ks_handles_empty_arrays():
    statistic, pvalue = compute_ks(np.array([]), np.array([1.0, 2.0, 3.0]))
    assert statistic == 0.0
    assert pvalue == 1.0
