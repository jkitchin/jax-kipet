"""
Tests for MixtureModel: linear-in-parameters blend property model.

The non-additive interaction motif of the project with no time axis: a property
of a blend is a sum of component contributions plus pairwise w_i w_j terms.
"""
import numpy as np
import pytest

from kipetax import MixtureModel


def test_design_matrix_shape_and_interaction_columns():
    mm = MixtureModel(n_components=3, interaction=True)
    W = np.array([[0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    G = mm.design(W)
    # 3 components + C(3,2)=3 pairs = 6 columns
    assert G.shape == (2, 6)
    # first blend: w0 w1 = 0.25, w0 w2 = 0, w1 w2 = 0
    assert G[0, 3] == pytest.approx(0.25)
    assert G[0, 4] == pytest.approx(0.0)

    add = MixtureModel(n_components=3, interaction=False)
    assert add.design(W).shape == (2, 3)


def test_additive_model_recovers_component_properties():
    """With no interaction in the data, the additive fit is exact."""
    rng = np.random.default_rng(0)
    props = np.array([1.0, 2.0, 5.0])          # scalar property per component
    W = rng.dirichlet(np.ones(3), size=20)
    Y = W @ props
    mm = MixtureModel(3, interaction=False).fit(W, Y)
    assert np.allclose(mm.component_props, props, atol=1e-9)


def test_interaction_term_is_recovered():
    """A single nonzero pairwise term is recovered; additive fit cannot see it."""
    rng = np.random.default_rng(1)
    props = np.array([1.0, 2.0, 5.0])
    d = np.array([0.0, 3.0, 0.0])              # only pair (0,2) interacts
    W = rng.dirichlet(np.ones(3), size=40)
    G = np.column_stack([W, W[:, 0] * W[:, 1], W[:, 0] * W[:, 2], W[:, 1] * W[:, 2]])
    Y = G @ np.concatenate([props, d])

    mm = MixtureModel(3, interaction=True).fit(W, Y)
    assert mm.interaction_parameter(0, 2) == pytest.approx(3.0, abs=1e-6)
    assert mm.interaction_parameter(0, 1) == pytest.approx(0.0, abs=1e-6)
    # symmetric access
    assert mm.interaction_parameter(2, 0) == mm.interaction_parameter(0, 2)

    add = MixtureModel(3, interaction=False).fit(W, Y)
    # the additive model buries the interaction: its component fit is biased
    assert not np.allclose(add.component_props, props, atol=1e-3)


def test_interaction_aware_predicts_better_on_interacting_data():
    rng = np.random.default_rng(2)
    props = np.log(np.array([100.0, 500.0, 2000.0]))
    d = np.array([0.0, 0.0, 2.0])              # pair (1,2) incompatible
    def gen(W):
        G = np.column_stack([W, W[:, 0] * W[:, 1], W[:, 0] * W[:, 2], W[:, 1] * W[:, 2]])
        return G @ np.concatenate([props, d])
    W_tr = rng.dirichlet(np.ones(3), size=30)
    W_te = rng.dirichlet(np.ones(3), size=30)
    Y_tr = gen(W_tr)
    Y_te = gen(W_te)

    add = MixtureModel(3, interaction=False).fit(W_tr, Y_tr)
    intr = MixtureModel(3, interaction=True).fit(W_tr, Y_tr)
    err_add = np.abs(add.predict(W_te) - Y_te).mean()
    err_int = np.abs(intr.predict(W_te) - Y_te).mean()
    assert err_int < err_add
    assert err_int < 1e-6


def test_infer_composition_inverts_a_spectrum():
    """Recover blend fractions from a rich (spectral) property."""
    rng = np.random.default_rng(3)
    n_feat = 30
    S = np.abs(rng.standard_normal((3, n_feat)))     # component spectra
    Gamma = np.zeros((3, n_feat))
    Gamma[2] = np.abs(rng.standard_normal(n_feat))   # pair (1,2) interaction band
    coef = np.vstack([S, Gamma])

    mm = MixtureModel(3, interaction=True)
    mm._coef = coef
    mm.component_props = S
    mm.interaction_props = Gamma

    w_true = np.array([0.2, 0.3, 0.5])
    spectrum = mm.predict(w_true)[0]
    w_rec = mm.infer_composition(spectrum)
    assert np.allclose(w_rec, w_true, atol=1e-3)
    assert w_rec.sum() == pytest.approx(1.0, abs=1e-6)


def test_infer_composition_rejects_a_matrix():
    mm = MixtureModel(2, interaction=False)
    mm.fit(np.eye(2), np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match='single property vector'):
        mm.infer_composition(np.zeros((3, 2)))
