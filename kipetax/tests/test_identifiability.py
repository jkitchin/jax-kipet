"""
Tests for analytic identifiability diagnostics
(``kipetax.calculation_tools.reduced_hessian``).

These tests exercise the Fisher Information Matrix / null-space machinery that
replaces the finite-difference covariance.  They evaluate the FIM at an
*analytically constructed feasible point* (the exact ODE solution placed on the
collocation grid) rather than at a re-solved NLP optimum.  This is both fast and
sufficient: structural (un)identifiability is a property of the constraint
Jacobian's rank, which is point-independent, so it does not require the
(intractable-at-test-speed) global NLP solve.  The well-identified case is a
sanity check; the rank-deficient case is the behaviour the module exists to fix
-- a null direction must be flagged, not hidden in an exploding interval.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.main_modules.reaction_model import ReactionModel
from kipetax.calculation_tools.reduced_hessian import (
    compute_fim,
    assess_identifiability,
    covariance_from_fim,
)


def _build_estimation(add_odes, k_values, t_end=5.0, n_points=15, nfe=10, ncp=3):
    """Build (builder, model, dae) for an A/B reaction with given parameters."""
    r = ReactionModel("id_test")
    params = {name: r.parameter(name, value=val, bounds=(0.01, 50))
              for name, val in k_values.items()}
    A = r.component("A", value=1.0)
    B = r.component("B", value=0.0)
    add_odes(r, params, A, B)
    t = np.linspace(0, t_end, n_points)
    df = pd.DataFrame({"A": np.exp(-t), "B": 1 - np.exp(-t)}, index=t)
    r.add_data(data=df)
    r.set_discretization(nfe=nfe, ncp=ncp)
    if r._times is None:
        r._infer_times_from_data()
    b = r._make_builder()
    m, dae, _ = b.build_parameter_estimation_model({})
    return b, m, dae


def _feasible_x_flat(m, dae, traj, k_values):
    """Flat primal vector with states set to the analytic trajectory."""
    cs = dae._cs
    tp = dae._element_points()
    state = {}
    for comp, fn in traj.items():
        arr = np.array([[fn(tp[i, k]) for k in range(cs.ncp + 1)]
                        for i in range(cs.nfe)])
        state[f"{cs.name}_{comp}"] = arr
    params = {f"P_{name}": np.array(val) for name, val in k_values.items()}
    parts = []
    for v in m._variables:
        if v.name in state:
            parts.append(state[v.name].ravel())
        elif v.name in params:
            parts.append(params[v.name].ravel())
        else:
            parts.append(np.zeros(v.size))
    return np.concatenate(parts)


def test_identifiable_model_full_rank():
    """A -> B with both species measured: k1 is identifiable (full-rank FIM)."""
    k = {"k1": 2.0}

    def odes(r, p, A, B):
        r.add_ode("A", -p["k1"] * A)
        r.add_ode("B", p["k1"] * A)

    b, m, dae = _build_estimation(odes, k)
    traj = {"A": lambda t: np.exp(-k["k1"] * t),
            "B": lambda t: 1.0 - np.exp(-k["k1"] * t)}
    x_flat = _feasible_x_flat(m, dae, traj, k)

    fim, names = compute_fim(b, m, dae, x_flat, {"A": 1e-4, "B": 1e-4})
    report = assess_identifiability(fim, names)

    assert names == ["k1"]
    assert report.rank == 1
    assert report.is_fully_identifiable
    assert report.identifiable["k1"] is True
    assert report.null_space == []
    assert report.warnings == []
    assert np.isfinite(report.condition_number)

    cov = covariance_from_fim(fim, report)
    assert np.isfinite(cov[0, 0]) and cov[0, 0] > 0


def test_structurally_unidentifiable_flagged():
    """A' = -(k1*k2)*A with only A measured: only the product k1*k2 is
    identifiable.  The FIM must be rank-deficient, the null direction must match
    the (k1, -k2) combination, the parameters must be flagged unidentifiable,
    and the variance must be infinite -- NOT a silent huge finite interval."""
    k = {"k1": 2.0, "k2": 1.5}
    prod = k["k1"] * k["k2"]

    def odes(r, p, A, B):
        r.add_ode("A", -(p["k1"] * p["k2"]) * A)
        r.add_ode("B", (p["k1"] * p["k2"]) * A)

    b, m, dae = _build_estimation(odes, k)
    traj = {"A": lambda t: np.exp(-prod * t),
            "B": lambda t: 1.0 - np.exp(-prod * t)}
    x_flat = _feasible_x_flat(m, dae, traj, k)

    # Only A is measured -> k1 and k2 collapse to their product.
    fim, names = compute_fim(b, m, dae, x_flat, {"A": 1e-4})
    report = assess_identifiability(fim, names)

    # Rank deficient by exactly one direction.
    assert report.rank == 1
    assert not report.is_fully_identifiable
    assert report.condition_number == float("inf")

    # Both parameters flagged unidentifiable.
    assert report.identifiable["k1"] is False
    assert report.identifiable["k2"] is False

    # The null direction is the (k1, -k2) combination.
    assert len(report.null_space) == 1
    combo = report.null_space[0]["combination"]
    null_vec = np.array([combo["k1"], combo["k2"]])
    null_vec = null_vec / np.linalg.norm(null_vec)
    expected = np.array([k["k1"], -k["k2"]])
    expected = expected / np.linalg.norm(expected)
    assert abs(null_vec @ expected) > 0.99

    # A warning is emitted naming the transient-excitation requirement.
    assert report.warnings
    assert any("transient excitation" in w for w in report.warnings)

    # Covariance reports infinite variance, not a meaningless huge finite value.
    cov = covariance_from_fim(fim, report)
    assert np.isinf(cov[0, 0])
    assert np.isinf(cov[1, 1])


class _FakeResult:
    """Minimal stand-in for a discopt SolveResult at a feasible point."""

    def __init__(self, x_dict, objective=1e-3):
        self.x = x_dict
        self.status = 'optimal'
        self.objective = objective


def _feasible_x_dict(m, dae, traj, k_values):
    """Same feasible point as _feasible_x_flat, keyed by variable name."""
    cs = dae._cs
    tp = dae._element_points()
    xd = {}
    state = {}
    for comp, fn in traj.items():
        state[f"{cs.name}_{comp}"] = np.array(
            [[fn(tp[i, k]) for k in range(cs.ncp + 1)] for i in range(cs.nfe)])
    params = {f"P_{name}": np.array(val) for name, val in k_values.items()}
    for v in m._variables:
        if v.name in state:
            xd[v.name] = state[v.name]
        elif v.name in params:
            xd[v.name] = params[v.name]
        else:
            xd[v.name] = np.zeros(v.shape) if v.shape else np.zeros(1)
    return xd


def test_parameter_estimator_wiring_flags_unidentifiable():
    """ParameterEstimator._compute_confidence_intervals routes through the
    analytic path and surfaces identifiability on the ResultsObject."""
    from kipetax.estimator_tools.parameter_estimator import ParameterEstimator
    from kipetax.estimator_tools.results_object import ResultsObject

    k = {"k1": 2.0, "k2": 1.5}
    prod = k["k1"] * k["k2"]

    def odes(r, p, A, B):
        r.add_ode("A", -(p["k1"] * p["k2"]) * A)
        r.add_ode("B", (p["k1"] * p["k2"]) * A)

    b, m, dae = _build_estimation(odes, k)
    traj = {"A": lambda t: np.exp(-prod * t),
            "B": lambda t: 1.0 - np.exp(-prod * t)}
    result = _FakeResult(_feasible_x_dict(m, dae, traj, k))

    pe = ParameterEstimator(b)
    ro = ResultsObject()
    ro.P = {"k1": 2.0, "k2": 1.5}
    ro.objective = result.objective

    pe._compute_confidence_intervals(ro, {"A": 1e-4}, m, dae, result)

    assert ro.cov_source == "reduced_hessian(rank-deficient)"
    assert ro.identifiability is not None
    assert ro.identifiability.rank == 1
    assert ro.confidence_intervals["k1"] == (float("-inf"), float("inf"))
    assert ro.confidence_intervals["k2"] == (float("-inf"), float("inf"))
    assert "RANK DEFICIENT" in str(ro)


def test_parameter_estimator_wiring_identifiable():
    """Wiring path yields finite CIs and full rank for an identifiable model."""
    from kipetax.estimator_tools.parameter_estimator import ParameterEstimator
    from kipetax.estimator_tools.results_object import ResultsObject

    k = {"k1": 2.0}

    def odes(r, p, A, B):
        r.add_ode("A", -p["k1"] * A)
        r.add_ode("B", p["k1"] * A)

    b, m, dae = _build_estimation(odes, k)
    traj = {"A": lambda t: np.exp(-k["k1"] * t),
            "B": lambda t: 1.0 - np.exp(-k["k1"] * t)}
    result = _FakeResult(_feasible_x_dict(m, dae, traj, k))

    pe = ParameterEstimator(b)
    ro = ResultsObject()
    ro.P = {"k1": 2.0}
    ro.objective = result.objective

    pe._compute_confidence_intervals(ro, {"A": 1e-4, "B": 1e-4}, m, dae, result)

    assert ro.cov_source == "reduced_hessian"
    assert ro.identifiability.is_fully_identifiable
    lo, hi = ro.confidence_intervals["k1"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo < 2.0 < hi


def test_report_summary_is_readable():
    """The report renders a human-readable summary with a rank line."""
    k = {"k1": 2.0}

    def odes(r, p, A, B):
        r.add_ode("A", -p["k1"] * A)
        r.add_ode("B", p["k1"] * A)

    b, m, dae = _build_estimation(odes, k)
    traj = {"A": lambda t: np.exp(-k["k1"] * t),
            "B": lambda t: 1.0 - np.exp(-k["k1"] * t)}
    x_flat = _feasible_x_flat(m, dae, traj, k)
    fim, names = compute_fim(b, m, dae, x_flat, {"A": 1e-4, "B": 1e-4})
    report = assess_identifiability(fim, names)

    text = report.summary()
    assert "FIM rank" in text
    assert "k1" in text
