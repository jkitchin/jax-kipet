"""
Tests for multiple experiments parameter estimation.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.main_modules.reaction_model import ReactionModel
from kipetax.main_modules.reaction_set import ReactionSet


def _make_ab_experiment(name, k1_true, t_end=5.0, n_points=15, noise=0.002, init_A=1.0):
    """Create a ReactionModel with synthetic A→B data for given k1_true."""
    rng = np.random.default_rng(hash(name) % (2**31))
    t = np.linspace(0, t_end, n_points)
    A_true = init_A * np.exp(-k1_true * t)
    B_true = init_A * (1 - np.exp(-k1_true * t))
    df = pd.DataFrame({
        'A': (A_true + rng.normal(0, noise, t.shape)).clip(0),
        'B': (B_true + rng.normal(0, noise, t.shape)).clip(0)
    }, index=t)

    r = ReactionModel(name)
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=init_A)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(data=df)
    r.set_discretization(nfe=20, ncp=3)
    return r


def test_reaction_set_add_model():
    """ReactionSet accepts ReactionModel instances."""
    rs = ReactionSet('test')
    r1 = _make_ab_experiment('exp1', k1_true=2.0)
    rs.add_model(r1)
    assert 'exp1' in [m.name for m in rs]


def test_reaction_set_add_non_model_raises():
    """ReactionSet raises TypeError for non-ReactionModel."""
    rs = ReactionSet('test')
    with pytest.raises(TypeError):
        rs.add_model("not_a_model")


@pytest.mark.slow
def test_two_experiments_shared_params():
    """Fitting two experiments with shared k1 recovers the true value."""
    k1_true = 2.0
    rs = ReactionSet('multi')
    rs.add_model(_make_ab_experiment('exp1', k1_true=k1_true, noise=0.001))
    rs.add_model(_make_ab_experiment('exp2', k1_true=k1_true, noise=0.001, init_A=0.5))

    ro = rs.run_opt(variances={'A': 1e-4, 'B': 1e-4})
    assert ro.status in ('optimal', 'feasible')
    assert 'k1' in ro.P
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.1, \
        f"k1 recovered={ro.P['k1']:.4f}, true={k1_true}"


@pytest.mark.slow
def test_multi_experiment_results_are_extracted_per_experiment():
    """Each member model gets back its own trajectories, not just the set."""
    rs = ReactionSet('multi_extract')
    r1 = _make_ab_experiment('exp1', k1_true=2.0, noise=0.001)
    r2 = _make_ab_experiment('exp2', k1_true=2.0, noise=0.001, init_A=0.5)
    rs.add_model(r1)
    rs.add_model(r2)

    ro = rs.run_opt(variances={'A': 1e-4, 'B': 1e-4})
    assert ro.status in ('optimal', 'feasible')
    assert set(ro.experiments) == {'exp1', 'exp2'}

    for name, model in (('exp1', r1), ('exp2', r2)):
        ri = ro.experiments[name]
        assert not ri.Z.empty
        assert list(ri.Z.columns) == ['A', 'B']
        # The member model is populated exactly as after a single-model run.
        assert model.results_dict['p_estimator'] is ri

    # Each experiment starts from its own A0, so the trajectories must differ.
    assert not np.isclose(ro.experiments['exp1'].Z['A'].iloc[0],
                          ro.experiments['exp2'].Z['A'].iloc[0])


# ── Spectral data across experiments ──

_WL = np.linspace(0.0, 100.0, 24)


def _gauss(center, width=8.0):
    return np.exp(-((_WL - center) ** 2) / (2.0 * width ** 2))


_S_TRUE = np.vstack([_gauss(25.0), _gauss(60.0)])       # (2 species, n_lambda)


def _make_ab_spectral(name, k1_true, seed, t_end=5.0, n_points=18, noise=1e-3):
    """A -> B experiment observed only through a Beer-Lambert D-matrix."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_end, n_points)
    C = np.column_stack([np.exp(-k1_true * t), 1.0 - np.exp(-k1_true * t)])
    D = pd.DataFrame(C @ _S_TRUE + rng.normal(0, noise, (len(t), len(_WL))),
                     index=t, columns=_WL)

    r = ReactionModel(name)
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A)
    r.add_data(name='spectral', data=D, category='spectral')
    r.set_discretization(nfe=15, ncp=3)
    return r


@pytest.mark.slow
def test_two_spectral_experiments_shared_param():
    """A shared rate constant is estimated from two spectral D-matrices.

    Combined estimation previously built objective terms from concentration
    data only, so a set of spectral experiments optimized an empty objective.
    """
    k1_true = 2.0
    rs = ReactionSet('multi_spectral')
    rs.add_model(_make_ab_spectral('expA', k1_true, seed=1))
    rs.add_model(_make_ab_spectral('expB', k1_true, seed=2, t_end=3.0))
    rs.set_shared_parameter('k1', value=1.0, bounds=(0.1, 10))

    ro = rs.run_opt(variances={'device': 1e-6})
    assert ro.status in ('optimal', 'feasible')
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.05, \
        f"k1 recovered={ro.P['k1']:.4f}, true={k1_true}"

    # Every experiment recovers its own S, on the right wavelength axis.
    for name in ('expA', 'expB'):
        S = ro.experiments[name].S
        assert S is not None
        assert S.shape == (len(_WL), 2)
        np.testing.assert_allclose(S.index.values, _WL)
        # Pure-component spectra peak where they were planted.
        assert abs(_WL[np.argmax(S['A'].values)] - 25.0) < 6.0
        assert abs(_WL[np.argmax(S['B'].values)] - 60.0) < 6.0


def _make_abc_spectral(name, k1_true, k2_true, seed, n_points=20, noise=1e-3):
    """A -> B -> C observed spectrally; k1 is shared, k2 is per-experiment."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 5.0, n_points)
    cA = np.exp(-k1_true * t)
    cB = k1_true / (k2_true - k1_true) * (np.exp(-k1_true * t)
                                          - np.exp(-k2_true * t))
    C = np.column_stack([cA, cB, 1.0 - cA - cB])
    S = np.vstack([_gauss(20.0), _gauss(50.0), _gauss(80.0)])
    D = pd.DataFrame(C @ S + rng.normal(0, noise, (len(t), len(_WL))),
                     index=t, columns=_WL)

    r = ReactionModel(name)
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    k2 = r.parameter('k2', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.component('C', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B', k1 * A - k2 * B)
    r.add_ode('C', k2 * B)
    r.add_data(name='spectral', data=D, category='spectral')
    r.set_discretization(nfe=15, ncp=3)
    return r


@pytest.mark.slow
def test_local_parameters_are_namespaced_per_experiment():
    """Only the declared parameter is shared; the rest stay per experiment.

    Declaring k1 shared must leave k2 free to differ between experiments, and
    the two k2 estimates must be reported separately rather than colliding on
    one key.
    """
    k1_true, k2_a, k2_b = 1.5, 0.6, 1.2
    rs = ReactionSet('multi_local')
    rs.add_model(_make_abc_spectral('expA', k1_true, k2_a, seed=5))
    rs.add_model(_make_abc_spectral('expB', k1_true, k2_b, seed=6))
    rs.set_shared_parameter('k1', value=1.0, bounds=(0.1, 10))

    ro = rs.run_opt(variances={'device': 1e-6})
    assert ro.status in ('optimal', 'feasible')

    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.05
    assert 'k2' not in ro.P, "a local parameter must not occupy a bare key"
    assert abs(ro.P['expA.k2'] - k2_a) / k2_a < 0.05, ro.P['expA.k2']
    assert abs(ro.P['expB.k2'] - k2_b) / k2_b < 0.05, ro.P['expB.k2']


def test_per_experiment_variance_override():
    """A nested dict overrides the flat variance for one experiment only."""
    rs = ReactionSet('vartest')
    variances = {'device': 1e-4, 'A': 1e-3, 'expB': {'device': 9e-9}}

    assert rs._experiment_variances(variances, 'expA') == {
        'device': 1e-4, 'A': 1e-3}
    assert rs._experiment_variances(variances, 'expB') == {
        'device': 9e-9, 'A': 1e-3}
