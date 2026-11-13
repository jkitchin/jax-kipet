"""
Regression tests: parameter covariance / confidence intervals for spectral
(category='spectral') data.

For a spectral fit D(t, lam) = C(theta) . S^T the pure-component spectra S are a
large block of free variables determined only by the least-squares objective,
not by any equality constraint. Both covariance paths used to mishandle this:

  * the analytic FIM treated S as constraint-determined free_z, making J_z
    rank-deficient and reporting Cov(theta) = inf (a spurious "unidentifiable");
  * the finite-difference path only formed concentration residuals, which are
    empty for spectral data, so it returned None.

The fix profiles S out (closed-form / VARPRO) in both paths. These tests pin the
expected behaviour: a finite, positive-definite Cov(theta) whose 95% CIs bracket
the true rate constants, and a full-rank identifiability report.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import odeint

from kipetax import ReactionModel

K1_TRUE, K2_TRUE = 2.0, 0.5


def _make_spectral_model():
    """A -> B -> C kinetics observed only through spectra D = C . S^T."""
    t = np.linspace(0, 8, 30)
    wl = np.arange(420, 601, 20.0)
    C = odeint(lambda y, _, a, b: [-a * y[0], a * y[0] - b * y[1], b * y[1]],
               [1, 0, 0], t, args=(K1_TRUE, K2_TRUE))
    S = np.column_stack([np.exp(-0.5 * ((wl - 460) / 35) ** 2) * 1.8,
                         np.exp(-0.5 * ((wl - 520) / 30) ** 2) * 1.4,
                         np.exp(-0.5 * ((wl - 575) / 35) ** 2) * 1.1])
    D = pd.DataFrame(C @ S.T + np.random.default_rng(0).normal(0, 1e-3, (len(t), len(wl))),
                     index=t, columns=wl)

    r = ReactionModel('inv')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    k2 = r.parameter('k2', value=1.0, bounds=(0.01, 5))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    Cc = r.component('C', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A - k2 * B)
    r.add_ode('C', k2 * B)
    r.add_data(name='spectral', data=D, category='spectral')
    r.set_discretization(nfe=25, ncp=3)
    return r


def _check_finite_pd_cov(ro):
    """Cov(theta) is a finite, symmetric, positive-definite 2x2 matrix."""
    assert ro.covariance is not None
    cov = np.asarray(ro.covariance, dtype=float)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov)), f"covariance not finite:\n{cov}"
    assert np.allclose(cov, cov.T, atol=1e-10)
    eig = np.linalg.eigvalsh(cov)
    assert np.all(eig > 0), f"covariance not positive-definite (eig={eig})"


def _check_cis_bracket_truth(ro):
    """Finite 95% CIs that bracket both true rate constants."""
    truth = {'k1': K1_TRUE, 'k2': K2_TRUE}
    for name, true_val in truth.items():
        lo, hi = ro.confidence_intervals[name]
        lo, hi = float(lo), float(hi)
        assert np.isfinite(lo) and np.isfinite(hi), f"{name} CI not finite: {(lo, hi)}"
        assert lo < ro.P[name] < hi, f"{name} estimate outside its own CI"
        assert lo < true_val < hi, f"{name} CI {(lo, hi)} does not bracket true {true_val}"


@pytest.mark.slow
def test_spectral_covariance_analytic():
    r = _make_spectral_model()
    r.run_opt(variances={'device': 1e-3}, covariance=True, cov_method='analytic')
    ro = r.results_dict['p_estimator']

    # Parameters are recovered.
    assert abs(ro.P['k1'] - K1_TRUE) < 0.2
    assert abs(ro.P['k2'] - K2_TRUE) < 0.1

    assert ro.cov_source == 'reduced_hessian'
    _check_finite_pd_cov(ro)
    _check_cis_bracket_truth(ro)

    # The kinetics ARE identifiable from this transient spectral data.
    report = ro.identifiability
    assert report is not None
    assert report.rank == 2
    assert report.is_fully_identifiable
    assert all(report.identifiable.values())


@pytest.mark.slow
def test_spectral_covariance_pounce():
    """POUNCE's reduced Hessian handles the spectral free-S block natively.

    The pinned reduced Hessian Schur-complements out every non-parameter
    variable (states and the free spectra S), so a spectral fit needs no special
    casing; the covariance should be finite, bracket truth, and match the
    analytic spectral path.
    """
    r = _make_spectral_model()
    r.run_opt(variances={'device': 1e-3}, covariance=True, cov_method='pounce')
    ro = r.results_dict['p_estimator']

    assert ro.cov_source == 'pounce_reduced_hessian'
    _check_finite_pd_cov(ro)
    _check_cis_bracket_truth(ro)

    r_an = _make_spectral_model()
    r_an.run_opt(variances={'device': 1e-3}, covariance=True, cov_method='analytic')
    se_p = np.sqrt(np.diag(np.asarray(ro.covariance, float)))
    se_an = np.sqrt(np.diag(np.asarray(
        r_an.results_dict['p_estimator'].covariance, float)))
    assert np.allclose(se_p, se_an, rtol=0.10), f"se_pounce={se_p} se_analytic={se_an}"


@pytest.mark.slow
def test_spectral_covariance_finite_difference():
    """The FD path forms a spectral residual and agrees with the analytic FIM.

    One ``cov_method='fd'`` run is expensive (it re-solves a simulation per
    perturbed parameter), so this single test covers both that the FD path
    yields a finite, bracketing covariance *and* that it agrees with the
    analytic reduced-Hessian path.
    """
    r_fd = _make_spectral_model()
    r_fd.run_opt(variances={'device': 1e-3}, covariance=True, cov_method='fd')
    ro_fd = r_fd.results_dict['p_estimator']

    assert ro_fd.cov_source == 'finite_difference'
    _check_finite_pd_cov(ro_fd)
    _check_cis_bracket_truth(ro_fd)

    # Agreement with the analytic (Kaufman VARPRO) covariance. Both trust the
    # supplied device variance (scale=1); the FD path differences the fully
    # profiled residual (Golub-Pereyra) with a re-fit S, so the two standard
    # errors are close but not identical.
    r_an = _make_spectral_model()
    r_an.run_opt(variances={'device': 1e-3}, covariance=True, cov_method='analytic')
    se_fd = np.sqrt(np.diag(np.asarray(ro_fd.covariance, dtype=float)))
    se_an = np.sqrt(np.diag(np.asarray(
        r_an.results_dict['p_estimator'].covariance, dtype=float)))
    assert np.allclose(se_fd, se_an, rtol=0.25), f"se_fd={se_fd} se_analytic={se_an}"
