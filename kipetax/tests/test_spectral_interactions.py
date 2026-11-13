"""
Tests for non-additive (interaction) spectral columns.

``ReactionModel.interaction(name, expr)`` adds a column to C whose value is an
algebraic expression of the states, carrying its own signed spectrum. It models
the terms that break Beer-Lambert bilinearity: adduct bands (~ c_A c_B) and
environment-driven band shifts (a derivative spectrum, hence signed).
"""
import numpy as np
import pandas as pd
import pytest

from kipetax import ReactionModel


K1_TRUE, K2_TRUE = 0.9, 0.5
G_STRENGTH = 1.6


def _spectra(nu):
    lor = lambda c, g, a: a * g ** 2 / ((nu - c) ** 2 + g ** 2)
    return (lor(520.0, 15.0, 1.0) + lor(1000.0, 20.0, 0.6),   # A
            lor(700.0, 18.0, 0.8) + lor(1350.0, 25.0, 0.7),   # B
            lor(900.0, 20.0, 0.9) + lor(1600.0, 22.0, 1.0),   # C
            lor(1150.0, 20.0, 0.7))                            # A.B adduct


def _concentrations(t):
    cA = np.exp(-K1_TRUE * t)
    cB = K1_TRUE / (K2_TRUE - K1_TRUE) * (np.exp(-K1_TRUE * t) - np.exp(-K2_TRUE * t))
    return cA, cB, 1.0 - cA - cB


def _data(interacting, n_nu=16, n_t=18, noise=2e-3, seed=0):
    nu = np.linspace(300.0, 1700.0, n_nu)
    t = np.linspace(0.0, 10.0, n_t)
    cA, cB, cC = _concentrations(t)
    SA, SB, SC, SAB = _spectra(nu)

    D = np.outer(cA, SA) + np.outer(cB, SB) + np.outer(cC, SC)
    if interacting:
        D = D + G_STRENGTH * np.outer(cA * cB, SAB)
    D = D + noise * np.random.default_rng(seed).standard_normal(D.shape)
    return pd.DataFrame(D, index=t, columns=nu), nu


def _model(interaction_aware, ridge=1e-3):
    r = ReactionModel('interaction_test')
    k1 = r.parameter('k1', value=0.6, bounds=(0.05, 3.0))
    k2 = r.parameter('k2', value=0.35, bounds=(0.05, 3.0))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    C = r.component('C', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A - k2 * B)
    r.add_ode('C', k2 * B)
    if interaction_aware:
        r.interaction('AB', A * B)
        r.set_spectral_regularization(ridge=ridge)
    r.set_discretization(nfe=12, ncp=3)
    return r


def test_interaction_registers_column_without_an_ode():
    """An interaction is algebraic: it must not become a differential state."""
    r = _model(interaction_aware=True)
    assert 'AB' in r._interactions
    assert 'AB' not in r.odes_dict, "interaction must not get an ODE"
    assert 'AB' not in r.components.names, "interaction must not be a component"


def test_ridge_setter():
    r = _model(interaction_aware=True, ridge=0.25)
    assert r._spectral_ridge == 0.25
    r.set_spectral_regularization()
    assert r._spectral_ridge == 0.0


@pytest.mark.slow
def test_interaction_column_removes_kinetic_bias():
    """Naive additive fit is biased on interacting data; the aware fit is not."""
    D, _ = _data(interacting=True)

    r_naive = _model(interaction_aware=False)
    r_naive.add_data(name='spectral', data=D, category='spectral')
    r_naive.run_opt(variances={'device': 4e-6})
    naive = r_naive.results_dict['p_estimator']

    r_aware = _model(interaction_aware=True)
    r_aware.add_data(name='spectral', data=D, category='spectral')
    r_aware.run_opt(variances={'device': 4e-6})
    aware = r_aware.results_dict['p_estimator']

    assert naive.status in ('optimal', 'feasible')
    assert aware.status in ('optimal', 'feasible')

    err_naive = abs(naive.P['k1'] - K1_TRUE) / K1_TRUE
    err_aware = abs(aware.P['k1'] - K1_TRUE) / K1_TRUE

    # The whole point: augmenting the forward model removes the bias.
    assert err_aware < err_naive, (
        f"interaction-aware fit ({err_aware:.3f}) should beat naive ({err_naive:.3f})")
    assert err_aware < 0.05


@pytest.mark.slow
def test_interaction_spectrum_is_recovered():
    """Gamma_AB comes back as an extra column of S, peaked at the adduct band."""
    D, nu = _data(interacting=True)

    r = _model(interaction_aware=True)
    r.add_data(name='spectral', data=D, category='spectral')
    r.run_opt(variances={'device': 4e-6})
    ro = r.results_dict['p_estimator']

    assert ro.S is not None
    assert list(ro.S.columns) == ['A', 'B', 'C', 'AB']

    gamma = ro.S['AB'].values
    truth = G_STRENGTH * _spectra(nu)[3]
    cos = float(gamma @ truth) / (np.linalg.norm(gamma) * np.linalg.norm(truth))
    assert cos > 0.95, f"recovered Gamma_AB cosine similarity {cos:.3f}"
    assert abs(nu[int(np.argmax(gamma))] - 1150.0) < 80.0
