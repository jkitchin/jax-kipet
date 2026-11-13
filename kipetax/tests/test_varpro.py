"""
Tests for the variable-projection (VARPRO) spectral estimator.

VARPRO eliminates the pure-component spectra S analytically (S* = C^+ D), so the
optimizer sees only the kinetic parameters and the cost is independent of the
number of wavelengths. These tests cover: recovery on a clean bilinear problem,
n_lambda-independence, the interaction-column path, and the label-ambiguity
resolution that a species-spectra non-negativity penalty provides.
"""
import time

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import odeint

from kipetax import ReactionModel


def _abc_spectral(k1_true=0.9, k2_true=0.5, n_lam=80, n_t=50, noise=2e-3,
                  interacting=False, g_strength=1.6, seed=0):
    nu = np.linspace(200.0, 1800.0, n_lam)
    t = np.linspace(0.0, 10.0, n_t)

    def lor(c, w, a):
        return a * w ** 2 / ((nu - c) ** 2 + w ** 2)
    SA = lor(520, 15, 1.0) + lor(1000, 20, 0.6)
    SB = lor(700, 18, 0.8) + lor(1350, 25, 0.7)
    SC = lor(900, 20, 0.9) + lor(1600, 22, 1.0)
    SAB = lor(1150, 20, 0.7)

    sol = odeint(lambda y, tt, a, b: [-a * y[0], a * y[0] - b * y[1], b * y[1]],
                 [1.0, 0.0, 0.0], t, args=(k1_true, k2_true))
    cA, cB, cC = sol[:, 0], sol[:, 1], sol[:, 2]
    D = np.outer(cA, SA) + np.outer(cB, SB) + np.outer(cC, SC)
    if interacting:
        D = D + g_strength * np.outer(cA * cB, SAB)
    D = D + noise * np.random.default_rng(seed).standard_normal(D.shape)
    return pd.DataFrame(D, index=t, columns=nu), nu, SAB


def _build(k1_0=0.3, k2_0=1.4, aware=False):
    r = ReactionModel('varpro')
    k1 = r.parameter('k1', value=k1_0, bounds=(0.05, 3.0))
    k2 = r.parameter('k2', value=k2_0, bounds=(0.05, 3.0))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    C = r.component('C', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A - k2 * B)
    r.add_ode('C', k2 * B)
    if aware:
        r.interaction('AB', A * B)
    return r


def test_varpro_recovers_rates_on_clean_data():
    D, _, _ = _abc_spectral()
    r = _build()
    r.add_data(name='sp', data=D, category='spectral')
    r.run_opt(variances={'device': 4e-6}, method='varpro')
    ro = r.results_dict['p_estimator']
    assert ro.status == 'optimal'
    assert ro.P['k1'] == pytest.approx(0.9, rel=0.03)
    assert ro.P['k2'] == pytest.approx(0.5, rel=0.05)


def test_varpro_resolves_the_k1_k2_label_ambiguity():
    """With free spectra the residual is symmetric in k1<->k2; the species
    non-negativity penalty must pick the physical assignment even from a start
    with the swapped ordering (k1 < k2)."""
    D, _, _ = _abc_spectral()
    r = _build(k1_0=0.3, k2_0=1.4)        # swapped ordering at the start
    r.add_data(name='sp', data=D, category='spectral')
    r.run_opt(variances={'device': 4e-6}, method='varpro')
    ro = r.results_dict['p_estimator']
    # k1 (A->B) must come back as the larger rate, not the swapped label.
    assert ro.P['k1'] == pytest.approx(0.9, rel=0.03)
    assert ro.P['k1'] > ro.P['k2']


def test_varpro_cost_is_independent_of_wavelength_count():
    """Doubling the spectral axis should not materially change the runtime,
    because S is projected out."""
    D1, _, _ = _abc_spectral(n_lam=100)
    D2, _, _ = _abc_spectral(n_lam=400)

    def timed(D):
        r = _build(k1_0=0.6, k2_0=0.35)
        r.add_data(name='sp', data=D, category='spectral')
        t0 = time.time()
        r.run_opt(variances={'device': 4e-6}, method='varpro')
        return time.time() - t0, r.results_dict['p_estimator']

    t1, ro1 = timed(D1)
    t2, ro2 = timed(D2)
    assert ro1.status == 'optimal' and ro2.status == 'optimal'
    assert ro2.P['k1'] == pytest.approx(0.9, rel=0.03)
    # 4x the wavelengths must not be 4x the time (allow generous slack for jit).
    assert t2 < 3.0 * t1 + 5.0


def test_varpro_recovers_the_interaction_spectrum():
    D, nu, SAB = _abc_spectral(interacting=True)
    r = _build(k1_0=0.6, k2_0=0.35, aware=True)
    r.add_data(name='sp', data=D, category='spectral')
    r.run_opt(variances={'device': 4e-6}, method='varpro')
    ro = r.results_dict['p_estimator']
    assert ro.status == 'optimal'
    assert ro.P['k1'] == pytest.approx(0.9, rel=0.04)
    assert 'AB' in ro.S.columns
    gamma = ro.S['AB'].values
    truth = 1.6 * SAB
    cos = float(gamma @ truth) / (np.linalg.norm(gamma) * np.linalg.norm(truth))
    assert cos > 0.95


def test_varpro_covariance_and_ci():
    D, _, _ = _abc_spectral(noise=1e-3)
    r = _build(k1_0=0.6, k2_0=0.35)
    r.add_data(name='sp', data=D, category='spectral')
    r.run_opt(variances={'device': 1e-6}, method='varpro', covariance=True)
    ro = r.results_dict['p_estimator']
    assert ro.cov_source == 'varpro_gauss_newton'
    assert ro.covariance.shape == (2, 2)
    lo, hi = ro.confidence_intervals['k1']
    assert np.isfinite(lo) and np.isfinite(hi) and lo < ro.P['k1'] < hi


def test_varpro_requires_spectral_data():
    t = np.linspace(0, 5, 10)
    df = pd.DataFrame({'A': np.exp(-2 * t)}, index=t)
    r = ReactionModel('nospec')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    r.add_ode('A', -k1 * A)
    r.add_data(data=df)
    with pytest.raises(ValueError, match='varpro'):
        r.run_opt(variances={'A': 1e-4}, method='varpro')


def test_varpro_and_collocation_agree_on_the_rates():
    """The two solvers should reach the same rate constants on the same data."""
    D, _, _ = _abc_spectral(n_lam=24, n_t=25)

    r_v = _build(k1_0=0.6, k2_0=0.35)
    r_v.add_data(name='sp', data=D, category='spectral')
    r_v.run_opt(variances={'device': 4e-6}, method='varpro')
    kv = r_v.results_dict['p_estimator'].P

    r_c = _build(k1_0=0.6, k2_0=0.35)
    r_c.set_discretization(nfe=15, ncp=3)
    r_c.add_data(name='sp', data=D, category='spectral')
    r_c.run_opt(variances={'device': 4e-6}, method='collocation')
    kc = r_c.results_dict['p_estimator'].P

    assert kv['k1'] == pytest.approx(kc['k1'], rel=0.05)
    assert kv['k2'] == pytest.approx(kc['k2'], rel=0.08)
