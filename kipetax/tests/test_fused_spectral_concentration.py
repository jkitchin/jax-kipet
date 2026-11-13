"""
Tests for fusing a spectral D-matrix with direct concentration measurements.

A spectral fit is invariant under ``c -> a*c, S -> S/a``, so when an initial
condition is itself unknown the absolute scale of a trajectory is free and any
rate constant multiplying a concentration inherits the ambiguity. A handful of
directly measured concentrations -- an offline GC endpoint, one calibrated
product channel -- pins it. Both objectives are weighted by 1/variance, so they
are commensurate and simply add.

Previously the two were alternatives: attaching spectral data made the builder
ignore every concentration dataset.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax import ReactionModel

WL = np.linspace(0.0, 100.0, 30)


def _gauss(center, width=8.0):
    return np.exp(-((WL - center) ** 2) / (2.0 * width ** 2))


S_TRUE = np.vstack([_gauss(25.0), _gauss(50.0), _gauss(75.0)])


def _abc(k1, k2, t):
    cA = np.exp(-k1 * t)
    cB = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    return np.column_stack([cA, cB, 1.0 - cA - cB])


def _spectral_abc(k1=1.5, k2=0.6, n_t=25, noise=1e-3, seed=0):
    t = np.linspace(0.0, 5.0, n_t)
    C = _abc(k1, k2, t)
    D = pd.DataFrame(C @ S_TRUE + np.random.default_rng(seed).normal(
        0, noise, (len(t), len(WL))), index=t, columns=WL)
    return D, t, C


def _model(name, D, conc=None, nfe=20):
    r = ReactionModel(name)
    k1 = r.parameter('k1', value=1.0, bounds=(0.05, 10.0))
    k2 = r.parameter('k2', value=1.0, bounds=(0.05, 10.0))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.component('C', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A - k2 * B)
    r.add_ode('C', k2 * B)
    r.add_data(name='spectral', data=D, category='spectral')
    if conc is not None:
        r.add_data(name='conc', data=conc)
    r.set_discretization(nfe=nfe, ncp=3)
    return r


def test_concentration_terms_are_added_to_a_spectral_objective():
    """The builder emits both blocks, rather than discarding one."""
    D, t, C = _spectral_abc()
    gc = pd.DataFrame({'C': [C[-1, 2]]}, index=[t[-1]])

    spectral_only = _model('s_only', D)._make_builder()
    fused = _model('fused', D, conc=gc)._make_builder()

    # The concentration dataset is visible to the builder ...
    assert spectral_only._get_concentration_data() == {}
    assert set(fused._get_concentration_data()) == {'C'}

    # ... and reaches the objective: the fused model contributes concentration
    # residuals even though spectral data is attached, where before the
    # spectral branch short-circuited them away.
    def n_terms(b):
        m, dae, _params = b.build_parameter_estimation_model(
            variances={'device': 1e-6, 'C': 1e-6})
        return len(b._build_concentration_objective(m, dae, {'C': 1e-6}))

    assert n_terms(spectral_only) == 0
    assert n_terms(fused) == 1


@pytest.mark.slow
def test_collocation_fused_fit_recovers_parameters():
    """Spectral + concentration collocation fit recovers k1, k2 and S."""
    k1_true, k2_true = 1.5, 0.6
    D, t, C = _spectral_abc(k1_true, k2_true)
    gc = pd.DataFrame({'C': [C[-1, 2]]}, index=[t[-1]])

    r = _model('fused_colloc', D, conc=gc)
    r.run_opt(variances={'device': 1e-6, 'C': 1e-6})
    ro = r.results_dict['p_estimator']

    assert ro.status in ('optimal', 'feasible')
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.05
    assert abs(ro.P['k2'] - k2_true) / k2_true < 0.05
    assert ro.S is not None and ro.S.shape == (len(WL), 3)


@pytest.mark.slow
def test_varpro_fused_fit_recovers_parameters():
    """The same fusion on the variable-projection path."""
    k1_true, k2_true = 1.5, 0.6
    D, t, C = _spectral_abc(k1_true, k2_true)
    gc = pd.DataFrame({'C': [C[-1, 2]]}, index=[t[-1]])

    r = _model('fused_varpro', D, conc=gc)
    r.run_opt(variances={'device': 1e-6, 'C': 1e-6}, method='varpro')
    ro = r.results_dict['p_estimator']

    assert ro.status == 'optimal'
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.05
    assert abs(ro.P['k2'] - k2_true) / k2_true < 0.05


@pytest.mark.slow
def test_varpro_without_concentration_data_is_unchanged():
    """Regression guard: the spectral-only varpro objective is untouched.

    With no concentration data and the default unit device variance the loss
    must be exactly the previous unweighted ``sum(R^2) + w*sum(neg^2)``.
    """
    k1_true, k2_true = 1.5, 0.6
    D, _t, _C = _spectral_abc(k1_true, k2_true)

    r = _model('spec_only_varpro', D)
    r.run_opt(method='varpro')
    ro = r.results_dict['p_estimator']

    assert ro.status == 'optimal'
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.05
    assert abs(ro.P['k2'] - k2_true) / k2_true < 0.05


@pytest.mark.slow
def test_a_tightly_weighted_concentration_point_is_honoured():
    """The concentration block actually pulls the fit, it is not inert.

    Weighted far more tightly than the spectra, a single measured endpoint must
    be reproduced by the fitted trajectory.
    """
    k1_true, k2_true = 1.5, 0.6
    D, t, C = _spectral_abc(k1_true, k2_true)
    c_end = C[-1, 2]
    gc = pd.DataFrame({'C': [c_end]}, index=[t[-1]])

    r = _model('anchored', D, conc=gc)
    r.run_opt(variances={'device': 1e-4, 'C': 1e-10})
    ro = r.results_dict['p_estimator']

    assert ro.status in ('optimal', 'feasible')
    assert abs(ro.Z['C'].iloc[-1] - c_end) / c_end < 0.02
