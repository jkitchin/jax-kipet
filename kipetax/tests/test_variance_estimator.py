"""
Tests for variance estimation.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.main_modules.reaction_model import ReactionModel


def _make_model_with_noisy_data(k1_true=2.0, sigma=0.01):
    """A → B model with known noise level."""
    t = np.linspace(0, 5, 20)
    rng = np.random.default_rng(42)
    A_true = np.exp(-k1_true * t) + rng.normal(0, sigma, t.shape)
    B_true = 1 - np.exp(-k1_true * t) + rng.normal(0, sigma, t.shape)
    df = pd.DataFrame({'A': A_true.clip(0), 'B': B_true.clip(0)}, index=t)

    r = ReactionModel('var_est')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=25, ncp=3)
    return r


@pytest.mark.slow
def test_variance_estimation_returns_dict():
    """Variance estimator returns a dict of component variances."""
    r = _make_model_with_noisy_data(sigma=0.01)
    variances = r.run_variance_estimator(max_iter=3, tol=1e-2)
    assert isinstance(variances, dict)
    assert 'A' in variances
    assert 'B' in variances
    assert all(v >= 0 for v in variances.values())


@pytest.mark.slow
def test_variance_order_of_magnitude():
    """Estimated variance should be within 2 orders of magnitude of true."""
    sigma_true = 0.01
    sigma_sq_true = sigma_true ** 2
    r = _make_model_with_noisy_data(sigma=sigma_true)
    variances = r.run_variance_estimator(max_iter=5, tol=1e-2)
    for comp, v in variances.items():
        ratio = v / sigma_sq_true
        assert 1e-2 <= ratio <= 1e2, \
            f"Variance for {comp}: estimated={v:.2e}, true={sigma_sq_true:.2e}, ratio={ratio:.2f}"


@pytest.mark.slow
def test_fixed_variance_skips_estimation():
    """Providing variances directly to run_opt skips variance estimation."""
    r = _make_model_with_noisy_data()
    variances = {'A': 1e-4, 'B': 1e-4}
    r.run_opt(variances=variances)
    ro = r.results_dict['p_estimator']
    assert ro.status in ('optimal', 'feasible')
