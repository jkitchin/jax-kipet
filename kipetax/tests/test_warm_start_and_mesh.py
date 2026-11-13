"""
Regression tests for NLP initialization and the graded collocation mesh.

``solve_local_nlp`` starts at the midpoint of the *clipped* bounds:
``0.5 * (clip(lb, -1e2, 1e2) + clip(ub, -1e2, 1e2))``. A concentration declared
with the default ``ub=1e20`` clips to 100 and therefore starts at 50, while the
true trajectory is O(1). Two distinct failures followed from that single bug:

1. Weighted by a small measurement variance (``1/sigma^2 = 2.5e5`` at
   ``sigma^2 = 4e-6``) the solver diverged and POUNCE reported UNBOUNDED -- on
   an objective that is a sum of squares and cannot be unbounded below.

2. On a stiff free-radical moment model it converged to a *wrong* rate constant
   while reporting ``status='optimal'``: a local minimum at ``kp = 1375`` when
   the true value is 1000. Refining the mesh did not help, because every mesh
   started from the same bad point.

Both are fixed by seeding the collocation blocks (constant trajectories at the
initial conditions, then a forward solve at the initial parameters).
"""
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from kipetax import ReactionModel
from kipetax.estimator_tools.parameter_estimator import ParameterEstimator


def _ab(variances, ub=None, nfe=30):
    t = np.linspace(0, 5, 21)
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {'A': np.clip(np.exp(-2.0 * t) + rng.normal(0, 2e-3, t.shape), 0, None),
         'B': np.clip(1 - np.exp(-2.0 * t) + rng.normal(0, 2e-3, t.shape), 0, None)},
        index=t)
    r = ReactionModel('ab')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    kw = {} if ub is None else {'bounds': (0.0, ub)}
    A = r.component('A', value=1.0, **kw)
    B = r.component('B', value=0.0, **kw)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=nfe, ncp=3)
    r.run_opt(variances=variances)
    return r.results_dict['p_estimator']


def test_small_variance_does_not_report_unbounded():
    """A sum-of-squares objective is bounded below; a tight variance must not
    make POUNCE report UNBOUNDED."""
    ro = _ab({'A': 4e-6, 'B': 4e-6})
    assert ro.status == 'optimal'
    assert ro.P['k1'] == pytest.approx(2.0, rel=0.05)


def test_wide_state_bounds_do_not_break_the_start():
    """Default ub=1e20 clips to 100 -> x0=50. The seed must override it."""
    for ub in (1e20, 500.0, 10.0):
        ro = _ab({'A': 4e-6, 'B': 4e-6}, ub=ub)
        assert ro.status == 'optimal', f"ub={ub} gave {ro.status}"
        assert ro.P['k1'] == pytest.approx(2.0, rel=0.05), f"ub={ub}"


def test_constant_state_guess_uses_initial_conditions():
    r = ReactionModel('seed')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.25)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A)
    r.set_time(5.0)
    r.set_discretization(nfe=7, ncp=3)

    guess = r._make_builder().constant_state_guess()
    assert set(guess) == {'t_A', 't_B'}
    assert guess['t_A'].shape == (7, 4)
    assert np.all(guess['t_A'] == 1.0)
    assert np.all(guess['t_B'] == 0.25)


def test_simulate_raises_instead_of_returning_empty_results():
    """A failed collocation solve must not hand back empty DataFrames."""
    r = ReactionModel('infeasible')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10), fixed=True)
    # The initial condition (1.0) lies outside the declared bounds, so the
    # collocation system has no solution.
    A = r.component('A', value=1.0, bounds=(5.0, 6.0))
    r.add_ode('A', -k1 * A)
    r.set_time(1.0)
    r.set_discretization(nfe=5, ncp=3)

    with pytest.raises(RuntimeError, match='Simulation failed'):
        r.simulate(solver_opts={'max_iter': 30})


def test_graded_mesh_matches_uniform_on_a_smooth_problem():
    """element_boundaries must reproduce the uniform answer when both resolve."""
    t = np.linspace(0, 5, 21)
    df = pd.DataFrame({'A': np.exp(-2.0 * t), 'B': 1 - np.exp(-2.0 * t)}, index=t)

    def fit(eb):
        r = ReactionModel('mesh')
        k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
        A = r.component('A', value=1.0)
        B = r.component('B', value=0.0)
        r.add_ode('A', -k1 * A)
        r.add_ode('B', k1 * A)
        r.add_data(data=df)
        if eb is None:
            r.set_discretization(nfe=20, ncp=3)
        else:
            r.set_discretization(ncp=3, element_boundaries=eb)
        r.run_opt(variances={'A': 1e-6, 'B': 1e-6})
        return r.results_dict['p_estimator']

    uniform = fit(None)
    graded = fit(np.concatenate([np.linspace(0, 1.0, 11),
                                 np.linspace(1.0, 5.0, 11)[1:]]))
    assert uniform.status == 'optimal' and graded.status == 'optimal'
    assert graded.P['k1'] == pytest.approx(uniform.P['k1'], rel=1e-3)


def test_late_starting_data_still_integrates_from_zero():
    """Initial conditions are declared at t=0, so inferring t0 from the first
    measurement silently solves a different model. Warn and use t0=0."""
    t = np.linspace(2.0, 5.0, 12)          # data starts late
    df = pd.DataFrame({'A': np.exp(-2.0 * t)}, index=t)

    r = ReactionModel('late')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)        # A(0) = 1, not A(2) = 1
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=20, ncp=3)

    with pytest.warns(UserWarning, match='initial conditions are declared at t=0'):
        r.run_opt(variances={'A': 1e-6})

    assert r._times[0] == 0.0
    ro = r.results_dict['p_estimator']
    assert ro.status == 'optimal'
    # With t0=0 honoured, the true rate is recoverable from late data alone.
    assert ro.P['k1'] == pytest.approx(2.0, rel=0.02)


def test_a_failed_status_raises_with_a_scaling_hint():
    """A non-optimal status must surface as a loud error, not an empty P dict.

    A weighted sum of squares is bounded below by zero, so an UNBOUNDED report
    is always an objective-scaling failure and the message has to say so. The
    status is injected rather than provoked: which weights actually break the
    solver is a property of the installed POUNCE, not a contract of kipetax
    (see :func:`test_extreme_measurement_weight_still_converges`), so
    triggering it through the solver makes the test track someone else's
    behaviour.
    """
    t = np.linspace(0.0, 5.0, 12)
    df = pd.DataFrame({'A': np.exp(-2.0 * t)}, index=t)
    r = ReactionModel('scaling')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    r.add_ode('A', -k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=20, ncp=3)

    from kipetax.estimator_tools.results_object import ResultsObject

    def fake_run_opt(self, **kwargs):
        ro = ResultsObject()
        ro.status = 'UNBOUNDED'
        return ro

    with mock.patch.object(ParameterEstimator, 'run_opt', fake_run_opt):
        with pytest.raises(RuntimeError, match='scaling failure'):
            r.run_opt(variances={'A': 1e-10})

    # The partial result is still reachable, as the message promises.
    assert r.results_dict['p_estimator'].status == 'UNBOUNDED'


def test_a_failed_status_without_unbounded_omits_the_scaling_hint():
    """The scaling hint is specific to UNBOUNDED; other failures must not
    claim a scaling cause they have no evidence for."""
    t = np.linspace(0.0, 5.0, 12)
    df = pd.DataFrame({'A': np.exp(-2.0 * t)}, index=t)
    r = ReactionModel('infeasible')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    r.add_ode('A', -k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=20, ncp=3)

    from kipetax.estimator_tools.results_object import ResultsObject

    def fake_run_opt(self, **kwargs):
        ro = ResultsObject()
        ro.status = 'INFEASIBLE'
        return ro

    with mock.patch.object(ParameterEstimator, 'run_opt', fake_run_opt):
        with pytest.raises(RuntimeError) as exc:
            r.run_opt(variances={'A': 1e-6})

    assert 'INFEASIBLE' in str(exc.value)
    assert 'scaling failure' not in str(exc.value)


@pytest.mark.slow
def test_extreme_measurement_weight_still_converges():
    """A very small variance is survivable, and must not silently misfit.

    This weight (1/sigma = 1e10) used to drive POUNCE to an UNBOUNDED report on
    an objective that cannot be unbounded below. It now converges, and the
    guard worth keeping is that it converges to the RIGHT answer rather than
    merely returning 'optimal'.
    """
    t = np.linspace(0.0, 5.0, 12)
    df = pd.DataFrame({'A': np.exp(-2.0 * t)}, index=t)
    r = ReactionModel('extreme_weight')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    r.add_ode('A', -k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=20, ncp=3)

    r.run_opt(variances={'A': 1e-10})
    ro = r.results_dict['p_estimator']

    assert ro.status in ('optimal', 'feasible')
    assert ro.P['k1'] == pytest.approx(2.0, rel=1e-3)


def test_data_starting_at_zero_does_not_warn():
    t = np.linspace(0.0, 5.0, 12)
    df = pd.DataFrame({'A': np.exp(-2.0 * t)}, index=t)
    r = ReactionModel('ontime')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    r.add_ode('A', -k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=10, ncp=3)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('error')           # any UserWarning becomes a failure
        r.run_opt(variances={'A': 1e-6})
    assert r._times == (0.0, 5.0)


@pytest.mark.slow
def test_stiff_moment_model_recovers_the_true_rate_constant():
    """Free-radical moments: radicals sit at ~1e-3 and relax in ~0.05 time
    units while the reaction runs for 10. From the bound-midpoint start this
    converged to kp=1375 with status='optimal'. The true value is 1000."""
    from scipy.integrate import solve_ivp

    KP, KT, KD, F = 1.0e3, 5.0e3, 0.5, 0.5
    M0, I0, TF = 1.0, 0.02, 10.0

    def rhs(t, y, kp):
        M, I, Y0, Y1, Y2, X0, X1, X2 = y
        RI = 2 * F * KD * I
        return [-kp * M * Y0, -KD * I, RI - 2 * KT * Y0 * Y0,
                RI + kp * M * Y0 - 2 * KT * Y0 * Y1,
                RI + kp * M * (2 * Y1 + Y0) - 2 * KT * Y0 * Y2,
                2 * KT * Y0 * Y0, 2 * KT * Y0 * Y1, 2 * KT * Y0 * Y2]

    sol = solve_ivp(rhs, (0, TF), [M0, I0, 0, 0, 0, 0, 0, 0.], args=(KP,),
                    method='LSODA', rtol=1e-10, atol=1e-13, dense_output=True).sol
    t_data = np.linspace(0.5, TF, 30)
    M_data = sol(t_data)[0] + 2e-3 * np.random.default_rng(0).standard_normal(30)

    r = ReactionModel('moments')
    kp = r.parameter('kp', value=5e2, bounds=(1e2, 5e3))
    kt = r.constant('kt', value=KT)
    kd = r.constant('kd', value=KD)
    f = r.constant('f', value=F)
    M = r.component('M', value=M0)
    I = r.component('I', value=I0)
    Y0 = r.state('Y0', value=0.0)
    Y1 = r.state('Y1', value=0.0)
    Y2 = r.state('Y2', value=0.0)
    X0 = r.state('X0', value=0.0)
    X1 = r.state('X1', value=0.0)
    X2 = r.state('X2', value=0.0)
    RI = 2 * f * kd * I
    r.add_ode('M', -kp * M * Y0)
    r.add_ode('I', -kd * I)
    r.add_ode('Y0', RI - 2 * kt * Y0 * Y0)
    r.add_ode('Y1', RI + kp * M * Y0 - 2 * kt * Y0 * Y1)
    r.add_ode('Y2', RI + kp * M * (2 * Y1 + Y0) - 2 * kt * Y0 * Y2)
    r.add_ode('X0', 2 * kt * Y0 * Y0)
    r.add_ode('X1', 2 * kt * Y0 * Y1)
    r.add_ode('X2', 2 * kt * Y0 * Y2)
    r.add_data(data=pd.DataFrame({'M': M_data}, index=t_data))
    r.set_discretization(nfe=40, ncp=3)

    r.run_opt(variances={'M': 4e-6}, solver_opts={'max_iter': 500})
    ro = r.results_dict['p_estimator']

    assert ro.status == 'optimal'
    assert ro.P['kp'] == pytest.approx(KP, rel=0.01), (
        f"kp={ro.P['kp']:.1f}; the 1375 local minimum is back")
