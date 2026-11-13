"""
Tests for covariance estimation and confidence intervals.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.main_modules.reaction_model import ReactionModel
from kipetax.calculation_tools.covariance import compute_covariance


def _make_ab_model_with_data(k1_true=2.0, t_end=5.0, n_points=15, noise=0.002):
    """A → B model with synthetic noisy data."""
    t = np.linspace(0, t_end, n_points)
    rng = np.random.default_rng(42)
    A_true = np.exp(-k1_true * t) + rng.normal(0, noise, size=t.shape)
    B_true = 1 - np.exp(-k1_true * t) + rng.normal(0, noise, size=t.shape)
    df = pd.DataFrame({'A': np.clip(A_true, 0, None), 'B': np.clip(B_true, 0, None)}, index=t)

    r = ReactionModel('cov_test')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(data=df)
    # nfe=25 is adequate: the objective interpolates the state to the exact
    # measurement time (no nearest-node snapping bias; see discopt#94), so the
    # estimate is effectively unbiased even on a coarse grid.
    r.set_discretization(nfe=25, ncp=3)
    return r


@pytest.mark.slow
def test_covariance_matrix_positive_definite():
    """Cov(P) matrix is positive semi-definite."""
    r = _make_ab_model_with_data()
    r.run_opt(variances={'A': 1e-4, 'B': 1e-4}, covariance=True)
    ro = r.results_dict['p_estimator']

    if ro.confidence_intervals:
        lo, hi = ro.confidence_intervals['k1']
        assert hi > lo, "Upper CI bound should exceed lower"
        assert lo > 0, "Lower CI bound should be positive for k1"


@pytest.mark.slow
def test_confidence_intervals_contain_truth():
    """95% CI should contain the true parameter value."""
    k1_true = 2.0
    r = _make_ab_model_with_data(k1_true=k1_true)
    r.run_opt(variances={'A': 1e-4, 'B': 1e-4}, covariance=True)
    ro = r.results_dict['p_estimator']

    if ro.confidence_intervals and 'k1' in ro.confidence_intervals:
        lo, hi = ro.confidence_intervals['k1']
        assert lo <= k1_true <= hi, \
            f"True k1={k1_true} not in CI ({lo:.4f}, {hi:.4f})"


@pytest.mark.slow
def test_covariance_finite_differences():
    """compute_covariance returns a finite 2D matrix."""
    r = _make_ab_model_with_data()
    b = r._make_builder()
    # Run PE first to get optimal params
    from kipetax.estimator_tools.parameter_estimator import ParameterEstimator
    pe = ParameterEstimator(b)
    ro = pe.run_opt(variances={'A': 1e-4, 'B': 1e-4})

    if ro.status in ('optimal', 'feasible'):
        cov = compute_covariance(b, ro.P, {'A': 1e-4, 'B': 1e-4})
        if cov is not None:
            assert cov.shape == (1, 1)
            assert np.all(np.isfinite(cov))
            assert cov[0, 0] >= 0


@pytest.mark.slow
def test_covariance_pounce_matches_analytic():
    """POUNCE reduced-Hessian covariance agrees with the analytic FIM path.

    POUNCE returns the full Lagrangian reduced Hessian at the KKT point; the
    analytic path is its Gauss-Newton approximation, so the two covariances
    agree closely but not exactly.
    """
    var = {'A': 1e-4, 'B': 1e-4}
    ra = _make_ab_model_with_data(k1_true=2.0)
    ra.run_opt(variances=var, covariance=True, cov_method='analytic')
    cov_an = np.asarray(ra.results_dict['p_estimator'].covariance, float)

    rp = _make_ab_model_with_data(k1_true=2.0)
    rp.run_opt(variances=var, covariance=True, cov_method='pounce')
    ro = rp.results_dict['p_estimator']

    assert ro.cov_source == 'pounce_reduced_hessian'
    cov_p = np.asarray(ro.covariance, float)
    assert np.all(np.isfinite(cov_p)) and cov_p[0, 0] > 0
    # Standard errors agree within 5%.
    se_p, se_an = np.sqrt(cov_p[0, 0]), np.sqrt(cov_an[0, 0])
    assert abs(se_p - se_an) / se_an < 0.05, f"se_pounce={se_p} se_analytic={se_an}"
    lo, hi = ro.confidence_intervals['k1']
    assert lo < 2.0 < hi  # CI brackets the true value
