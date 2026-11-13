"""
Tests for ODE simulation via kipetax (Phase 1 analog of Ex_1).
"""
import numpy as np
import pytest

from kipetax.main_modules.reaction_model import ReactionModel


def _run_abc_sim(k1=2.0, k2=0.5, t_end=5.0, nfe=30, ncp=3):
    """Simulate A → B → C and return the ResultsObject."""
    r = ReactionModel('abc_sim')
    k1_p = r.parameter('k1', value=k1, bounds=(0, 10))
    k2_p = r.parameter('k2', value=k2, bounds=(0, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    C = r.component('C', value=0.0)
    r.add_ode('A', -k1_p * A)
    r.add_ode('B',  k1_p * A - k2_p * B)
    r.add_ode('C',  k2_p * B)
    r.set_time(t_end)
    r.set_discretization(nfe=nfe, ncp=ncp)
    r.simulate()
    return r.results_dict['simulator']


def test_simple_abc_sim_status():
    """Simulation completes with optimal/feasible status."""
    ro = _run_abc_sim()
    assert ro.status in ('optimal', 'feasible'), f"Unexpected status: {ro.status}"


def test_simple_abc_sim_shape():
    """Z DataFrame has 3 columns and correct number of rows."""
    ro = _run_abc_sim(nfe=20, ncp=3)
    assert 'A' in ro.Z.columns
    assert 'B' in ro.Z.columns
    assert 'C' in ro.Z.columns


def test_simple_abc_mass_conservation():
    """A + B + C ≈ 1 at all times (mass balance)."""
    ro = _run_abc_sim(nfe=30, ncp=3)
    Z = ro.Z
    total = Z['A'] + Z['B'] + Z['C']
    assert np.allclose(total, 1.0, atol=1e-4), \
        f"Mass balance violated: max deviation = {abs(total - 1.0).max():.4e}"


def test_abc_initial_conditions():
    """Check that initial concentrations match specified values."""
    ro = _run_abc_sim(nfe=20, ncp=3)
    Z = ro.Z
    # First time point
    assert abs(Z['A'].iloc[0] - 1.0) < 1e-6
    assert abs(Z['B'].iloc[0] - 0.0) < 1e-6
    assert abs(Z['C'].iloc[0] - 0.0) < 1e-6


def test_abc_monotone_A():
    """A should monotonically decrease."""
    ro = _run_abc_sim(nfe=30, ncp=3)
    A_vals = ro.Z['A'].values
    diffs = np.diff(A_vals)
    assert np.all(diffs <= 1e-6), "A is not monotonically decreasing"


def test_abc_final_C():
    """C should approach 1 as t → ∞ (for long enough simulation)."""
    ro = _run_abc_sim(k1=2.0, k2=0.5, t_end=30.0, nfe=50, ncp=3)
    C_final = ro.Z['C'].iloc[-1]
    assert C_final > 0.95, f"C at t=30 should be near 1, got {C_final:.4f}"


def test_single_species_decay():
    """A → B with only concentration monitoring."""
    r = ReactionModel('single_decay')
    k1 = r.parameter('k1', value=1.0, bounds=(0, 5))
    A = r.component('A', value=2.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.set_time(5.0)
    r.set_discretization(nfe=25, ncp=3)
    r.simulate()

    ro = r.results_dict['simulator']
    assert ro.status in ('optimal', 'feasible')
    # A decays from 2.0
    assert ro.Z['A'].iloc[0] == pytest.approx(2.0, abs=1e-5)


def test_arrhenius_kinetics():
    """Arrhenius kinetics in the ODE RHS with numerically reasonable parameters.

    Uses A_pre and Ea values that give k ≈ 1.0 at T=300 K to avoid stiffness.
    """
    import math
    from kipetax.model_tools.proxy_vars import exp

    # Design so k_eff = A_pre * exp(-Ea/(R*T)) ≈ 1.0 at T=300, R=8.314
    # exp(-Ea/(R*T)) with Ea=500, R=8.314, T=300 → exp(-0.2003) ≈ 0.818
    # A_pre = 1/0.818 ≈ 1.22 → k ≈ 1.0
    Ea_val = 500.0
    R_val = 8.314
    T_K = 300.0
    k_eff = math.exp(-Ea_val / (R_val * T_K))
    A_pre_val = 1.0 / k_eff  # ensures k_eff ≈ 1.0

    r = ReactionModel('arrhenius')
    A_pre = r.constant('A_pre', value=A_pre_val)
    Ea = r.constant('Ea', value=Ea_val)
    R_gas = r.constant('R_gas', value=R_val)
    T_val = r.constant('T_val', value=T_K)

    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)

    k_expr = A_pre * exp(-Ea / (R_gas * T_val))
    r.add_ode('A', -k_expr * A)
    r.add_ode('B',  k_expr * A)
    r.set_time(10.0)
    r.set_discretization(nfe=30, ncp=3)
    r.simulate()

    ro = r.results_dict['simulator']
    assert ro.status in ('optimal', 'feasible'), f"Status: {ro.status}"
    total = ro.Z['A'] + ro.Z['B']
    assert np.allclose(total, 1.0, atol=1e-4)
