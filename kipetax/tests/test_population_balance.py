"""
Tests for PopulationBalanceModel: a distributed state fit by sequential shooting.

These cover the mechanics (integration schemes, scalar-state bookkeeping,
convergence classification) and one end-to-end recovery on a small growth PBE,
the class of problem ReactionModel cannot express.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from kipetax import PopulationBalanceModel


def test_pure_advection_conserves_and_shifts():
    """Upwind Euler on a pulse: it advects at the growth speed and does not
    create mass in the interior."""
    K = 30
    grid = np.linspace(0.0, 3.0, K)
    dx = grid[1] - grid[0]
    speed = 0.5

    def rhs(N, t, p):
        g = p['g']
        F = g * N
        dN = -F / dx
        dN = dN.at[1:].add(F[:-1] / dx)
        return dN

    N0 = np.exp(-((grid - 0.5) / 0.2) ** 2)
    pbm = PopulationBalanceModel('adv')
    pbm.set_coordinate('x', grid)
    pbm.parameter('g', value=speed, bounds=(1e-3, 5.0))
    pbm.set_initial_state(jnp.asarray(N0))
    pbm.set_dynamics(rhs, t_end=2.0, nt=400, scheme='euler')

    ro = pbm.simulate()
    final = ro.distribution()
    peak_shift_cells = int(np.argmax(final)) - int(np.argmax(N0))
    expected = speed * 2.0 / dx
    assert abs(peak_shift_cells - expected) <= 2
    # No new mass appears in the interior (only advective outflow at the top).
    assert final.sum() <= N0.sum() + 1e-8


def test_scalar_state_is_carried():
    """A scalar appended to the distribution integrates independently."""
    K = 5
    grid = np.arange(K, dtype=float)

    def rhs(state, t, p):
        # distribution frozen; scalar grows at a constant rate
        dN = jnp.zeros(K)
        dQ = p['rate']
        return jnp.concatenate([dN, jnp.array([dQ])])

    pbm = PopulationBalanceModel('scalar')
    pbm.parameter('rate', value=2.0, bounds=(1e-3, 10.0))
    pbm.set_initial_state(jnp.concatenate([jnp.ones(K), jnp.array([0.0])]))
    pbm.set_dynamics(rhs, t_end=3.0, nt=300, scheme='rk4')

    ro = pbm.simulate()
    assert ro.state_history[-1, K] == pytest.approx(2.0 * 3.0, rel=1e-4)
    assert np.allclose(ro.state_history[-1, :K], 1.0)


def test_log_space_requires_positive_bounds():
    pbm = PopulationBalanceModel('bad')
    with pytest.raises(ValueError, match='log space'):
        pbm.parameter('k', value=1.0, bounds=(0.0, 10.0))


def test_nnls_inversion_recovers_a_known_combination():
    """The naive baseline inverts a spectrum onto a known kernel."""
    kernel = np.abs(np.random.default_rng(0).standard_normal((20, 4)))
    true = np.array([[1.0, 0.0, 2.0, 0.5]])
    spectra = true @ kernel.T
    rec = PopulationBalanceModel.nnls_inversion(spectra, kernel)
    assert rec.shape == (1, 4)
    assert np.allclose(rec, true, atol=1e-6)


def test_growth_pbe_recovers_rate_and_beats_naive_inversion():
    """End to end: fit a growth rate to a spectral time series through the
    mechanism, and confirm it beats a per-spectrum inversion."""
    K = 30
    R = np.linspace(0.8, 5.0, K)
    DR = float(R[1] - R[0])
    LAM = np.linspace(400.0, 700.0, 60)
    M0, MEQ, CVOL = 1.0, 0.05, 0.6
    E_BULK, A_CONF, WIDTH = 1.7, 0.9, 30.0
    TF, NT = 20.0, 800
    KG_TRUE = 0.02
    JSTAR = 1

    LPEAK = 1240.0 / (E_BULK + A_CONF / R ** 2)
    KERNEL = (R ** 3)[None, :] * np.exp(-(LAM[:, None] - LPEAK[None, :]) ** 2
                                        / (2 * WIDTH ** 2))
    T_OBS = np.array([2.0, 4.0, 7.0, 11.0, 16.0, 20.0])
    IDX = np.clip((T_OBS / TF * NT).astype(int), 0, NT - 1)

    _R, _KERNEL, _IDX = jnp.asarray(R), jnp.asarray(KERNEL), jnp.asarray(IDX)

    def step(N, t, p):
        m = jnp.maximum(M0 - CVOL * jnp.sum(N * _R ** 3) * DR, MEQ)
        S = m / MEQ
        G = jnp.where(S > 1.0, p['kg'] / _R * (S - 1.0), 0.0)
        F = G * N
        dN = -F / DR
        dN = dN.at[1:].add(F[:-1] / DR)
        dN = dN.at[JSTAR].add(0.02 / DR)      # fixed nucleation source
        return dN

    def observe(hist, p):
        return (hist[_IDX] * DR) @ _KERNEL.T

    def build():
        pbm = PopulationBalanceModel('growth')
        pbm.set_coordinate('r', R)
        pbm.parameter('kg', value=0.01, bounds=(1e-3, 0.1))
        pbm.set_initial_state(jnp.zeros(K))
        pbm.set_dynamics(step, t_end=TF, nt=NT, scheme='euler')
        pbm.set_observation(observe)
        return pbm

    truth = build().simulate({'kg': KG_TRUE})
    A_true = np.asarray(observe(jnp.asarray(truth.state_history), {'kg': KG_TRUE}))
    A_data = A_true + 0.02 * A_true.max() * np.random.default_rng(0).standard_normal(A_true.shape)

    pbm = build()
    pbm.add_data(A_data)
    ro = pbm.run_opt()

    assert ro.status == 'optimal'
    assert ro.P['kg'] == pytest.approx(KG_TRUE, rel=0.05)

    # Mechanism reconstruction beats naive per-spectrum inversion on shape.
    N_true = np.asarray(truth.state_history)[IDX]
    N_fit = ro.state_history[IDX]
    N_naive = PopulationBalanceModel.nnls_inversion(A_data, KERNEL * DR)

    def final_shape(N):
        return N[-1] / max(N[-1].sum(), 1e-12)
    l1_fit = np.abs(final_shape(N_fit) - final_shape(N_true)).sum()
    l1_naive = np.abs(final_shape(N_naive) - final_shape(N_true)).sum()
    assert l1_fit < l1_naive
