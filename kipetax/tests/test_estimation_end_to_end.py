"""
End-to-end parameter estimation through the local NLP solver, with analytic
identifiability diagnostics computed at the *genuine* optimum.

These exercise the full pipeline:
``ReactionModel.run_opt`` -> local POUNCE solve -> ``analytic_covariance`` ->
``ResultsObject``.  They run a real (local) NLP solve, so they take ~1 s each
rather than the intractable global branch-and-bound path.

The equivalence test cross-validates the analytic Fisher-Information covariance
against an independent ``scipy.optimize.curve_fit`` on the closed-form model
``A(t) = exp(-k1 t)`` -- a method that shares no code with kipetax -- confirming
the reduced-Hessian standard error is correct, not merely self-consistent.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.main_modules.reaction_model import ReactionModel


def _ab_model(k1_true=2.0, t_end=5.0, n_points=21, noise=2e-3, nfe=30, ncp=3,
              measure=('A', 'B'), seed=1):
    t = np.linspace(0, t_end, n_points)
    rng = np.random.default_rng(seed)
    data = {}
    if 'A' in measure:
        data['A'] = np.clip(np.exp(-k1_true * t) + rng.normal(0, noise, t.shape), 0, None)
    if 'B' in measure:
        data['B'] = np.clip(1 - np.exp(-k1_true * t) + rng.normal(0, noise, t.shape), 0, None)
    df = pd.DataFrame(data, index=t)

    r = ReactionModel('e2e')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=nfe, ncp=ncp)
    return r, df, t


def test_end_to_end_local_solve_identifiable():
    """Full run_opt on A->B: solves locally, recovers k1, reports a finite CI
    and a full-rank, reduced-Hessian covariance."""
    k1_true = 2.0
    r, df, t = _ab_model(k1_true=k1_true)
    variances = {'A': 4e-6, 'B': 4e-6}
    r.run_opt(variances=variances, covariance=True)
    ro = r.results_dict['p_estimator']

    assert ro.status == 'optimal'
    assert 'k1' in ro.P
    # local solve recovers the true rate (collocation is fine at nfe=30)
    assert ro.P['k1'] == pytest.approx(k1_true, rel=0.05)

    assert ro.cov_source == 'reduced_hessian'
    assert ro.identifiability.is_fully_identifiable
    assert ro.identifiability.rank == 1

    lo, hi = ro.confidence_intervals['k1']
    assert np.isfinite(lo) and np.isfinite(hi) and lo < ro.P['k1'] < hi


def test_end_to_end_unidentifiable_under_steady_state():
    """A reaction measured at only one (early) point cannot pin the rate well;
    measuring only the product combination flags it.  Here we drive the
    structural case: A' = -(k1*k2) A with only A measured -> rank deficient,
    infinite CIs, transient-excitation warning -- end to end through run_opt."""
    t = np.linspace(0, 5, 21)
    rng = np.random.default_rng(2)
    prod = 2.0 * 1.5
    df = pd.DataFrame({'A': np.clip(np.exp(-prod * t) + rng.normal(0, 2e-3, t.shape), 0, None)},
                      index=t)
    r = ReactionModel('e2e_un')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    k2 = r.parameter('k2', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -(k1 * k2) * A)
    r.add_ode('B', (k1 * k2) * A)
    r.add_data(data=df)
    r.set_discretization(nfe=30, ncp=3)
    r.run_opt(variances={'A': 4e-6}, covariance=True)
    ro = r.results_dict['p_estimator']

    assert ro.status == 'optimal'
    assert ro.cov_source == 'reduced_hessian(rank-deficient)'
    assert ro.identifiability.rank == 1
    assert not ro.identifiability.is_fully_identifiable
    assert ro.confidence_intervals['k1'] == (float('-inf'), float('inf'))
    assert ro.confidence_intervals['k2'] == (float('-inf'), float('inf'))
    assert any('transient excitation' in w for w in ro.identifiability.warnings)


def test_analytic_se_matches_independent_curve_fit():
    """Cross-validate the reduced-Hessian standard error against an independent
    scipy.optimize.curve_fit on the closed-form A(t) = exp(-k1 t).

    To compare like with like, kipetax measures only A (the species scipy fits),
    with the same data and the same known noise variance.  The two standard
    errors -- one from kipetax's analytic FIM, one from scipy's covariance --
    should agree closely; they encode the same Fisher information."""
    from scipy.optimize import curve_fit

    k1_true = 2.0
    noise = 2e-3
    sigma_sq = noise ** 2
    r, df, t = _ab_model(k1_true=k1_true, noise=noise, nfe=40, measure=('A',))

    r.run_opt(variances={'A': sigma_sq}, covariance=True)
    ro = r.results_dict['p_estimator']
    assert ro.status == 'optimal'
    assert ro.identifiability.is_fully_identifiable
    se_kipet = float(np.sqrt(ro.covariance[0, 0]))
    k1_kipet = ro.P['k1']

    # Independent reference fit on the same A data with the same noise level.
    a_data = df['A'].values
    popt, pcov = curve_fit(lambda tt, k: np.exp(-k * tt), t, a_data,
                           p0=[1.0], sigma=np.full_like(a_data, noise),
                           absolute_sigma=True)
    k1_scipy = float(popt[0])
    se_scipy = float(np.sqrt(pcov[0, 0]))

    # Point estimates agree (collocation vs exact integration).
    assert k1_kipet == pytest.approx(k1_scipy, rel=0.05)
    # Standard errors encode the same information -> agree closely.
    assert se_kipet == pytest.approx(se_scipy, rel=0.15)
