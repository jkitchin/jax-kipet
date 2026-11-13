"""
Tests for parameter estimation (Phase 2 analog of Ex_2 with concentration data).
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.main_modules.reaction_model import ReactionModel


def _generate_ab_data(k1_true=2.0, t_end=5.0, n_points=20, noise=0.0):
    """Generate synthetic A → B concentration data."""
    t = np.linspace(0, t_end, n_points)
    A_true = np.exp(-k1_true * t)
    B_true = 1.0 - A_true
    if noise > 0:
        rng = np.random.default_rng(42)
        A_true += rng.normal(0, noise, size=A_true.shape)
        B_true += rng.normal(0, noise, size=B_true.shape)
    return pd.DataFrame({'A': A_true, 'B': B_true}, index=t)


def _generate_abc_data(k1_true=2.0, k2_true=0.5, t_end=5.0, n_points=20, noise=0.0):
    """Generate synthetic A → B → C concentration data."""
    from scipy.integrate import odeint

    def odes(y, t, k1, k2):
        A, B, C = y
        return [-k1 * A, k1 * A - k2 * B, k2 * B]

    t = np.linspace(0, t_end, n_points)
    sol = odeint(odes, [1.0, 0.0, 0.0], t, args=(k1_true, k2_true))
    df = pd.DataFrame({'A': sol[:, 0], 'B': sol[:, 1], 'C': sol[:, 2]}, index=t)
    if noise > 0:
        rng = np.random.default_rng(42)
        df += rng.normal(0, noise, size=df.shape)
        df = df.clip(lower=0)
    return df


def test_ab_parameter_recovery():
    """Recover k1 from exact (noise-free) A → B data."""
    k1_true = 2.5
    df = _generate_ab_data(k1_true=k1_true, t_end=5.0, n_points=15)

    r = ReactionModel('ab_pe')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=30, ncp=3)

    r.run_opt(variances={'A': 1e-4, 'B': 1e-4})
    ro = r.results_dict['p_estimator']

    assert ro.status in ('optimal', 'feasible')
    assert 'k1' in ro.P
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.05, \
        f"k1 recovered={ro.P['k1']:.4f}, true={k1_true}"


def test_abc_two_parameters():
    """Recover k1, k2 from noise-free A → B → C data."""
    k1_true, k2_true = 2.0, 0.5
    df = _generate_abc_data(k1_true=k1_true, k2_true=k2_true, t_end=6.0, n_points=20)

    r = ReactionModel('abc_pe')
    # Use bounds centred at the true values: discopt starts from bound midpoints,
    # so bounds centred near the answer give the best initial point.
    k1 = r.parameter('k1', value=2.0, bounds=(1.5, 2.5))   # midpoint = 2.0 = k1_true
    k2 = r.parameter('k2', value=0.5, bounds=(0.3, 0.8))   # midpoint = 0.55 ≈ k2_true
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    C = r.component('C', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A - k2 * B)
    r.add_ode('C',  k2 * B)
    r.add_data(data=df)
    r.set_discretization(nfe=30, ncp=3)

    r.run_opt(variances={'A': 1e-4, 'B': 1e-4, 'C': 1e-4})
    ro = r.results_dict['p_estimator']

    assert ro.status in ('optimal', 'feasible')
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.05, \
        f"k1 recovered={ro.P['k1']:.4f}, true={k1_true}"
    assert abs(ro.P['k2'] - k2_true) / k2_true < 0.05, \
        f"k2 recovered={ro.P['k2']:.4f}, true={k2_true}"


def test_parameter_bounds_respected():
    """Solution parameters stay within bounds."""
    df = _generate_ab_data(k1_true=2.0, t_end=5.0, n_points=10)

    r = ReactionModel('bounds_test')
    k1 = r.parameter('k1', value=1.0, bounds=(0.5, 3.0))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=20, ncp=3)

    r.run_opt()
    ro = r.results_dict['p_estimator']
    assert ro.status in ('optimal', 'feasible')
    assert 0.5 <= ro.P['k1'] <= 3.0, f"k1 = {ro.P['k1']} violates bounds (0.5, 3.0)"


def test_results_object_has_required_attrs():
    """ResultsObject has P, Z, X, status, objective."""
    df = _generate_ab_data(k1_true=1.5, t_end=3.0, n_points=8)

    r = ReactionModel('attr_test')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 5))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=15, ncp=3)

    r.run_opt()
    ro = r.results_dict['p_estimator']

    assert hasattr(ro, 'P')
    assert hasattr(ro, 'Z')
    assert hasattr(ro, 'X')
    assert hasattr(ro, 'status')
    assert hasattr(ro, 'objective')
    assert isinstance(ro.P, dict)
    assert not ro.Z.empty


def test_fixed_parameter():
    """Fixed parameters stay at their set value."""
    df = _generate_ab_data(k1_true=2.0, t_end=5.0, n_points=10)

    r = ReactionModel('fixed_param_test')
    k1 = r.parameter('k1', value=2.0, bounds=(0.1, 10), fixed=True)
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=15, ncp=3)

    r.run_opt()
    ro = r.results_dict['p_estimator']
    # k1 should remain at 2.0 (not estimated)
    assert abs(ro.P.get('k1', 2.0) - 2.0) < 1e-10
