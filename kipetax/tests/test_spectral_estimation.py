"""
Tests for spectral (Beer-Lambert) parameter estimation.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.main_modules.reaction_model import ReactionModel


def _generate_spectral_data(k1_true=1.5, t_end=5.0, n_times=15, wavelengths=None,
                            S_true=None, noise=0.001):
    """
    Generate synthetic D matrix: D[t, lam] = sum_c C[t,c] * S[lam,c]
    """
    from scipy.integrate import odeint

    if wavelengths is None:
        wavelengths = list(range(400, 601, 25))  # 9 wavelengths

    n_lam = len(wavelengths)
    n_comp = 2  # A and B

    if S_true is None:
        # Gaussian S matrices
        lam_arr = np.array(wavelengths, dtype=float)
        S_A = np.exp(-0.5 * ((lam_arr - 450) / 30) ** 2) * 2.0
        S_B = np.exp(-0.5 * ((lam_arr - 550) / 30) ** 2) * 1.5
        S_true = np.column_stack([S_A, S_B])

    t = np.linspace(0, t_end, n_times)

    def odes(y, t, k1):
        return [-k1 * y[0], k1 * y[0]]

    sol = odeint(odes, [1.0, 0.0], t, args=(k1_true,))
    # D = C @ S^T
    D = sol @ S_true.T

    if noise > 0:
        rng = np.random.default_rng(42)
        D += rng.normal(0, noise, size=D.shape)

    D_df = pd.DataFrame(D, index=t, columns=wavelengths)
    S_df = pd.DataFrame(S_true, index=wavelengths, columns=['A', 'B'])
    return D_df, S_df


@pytest.mark.slow
def test_fit_with_spectral_data():
    """Recover k1 from synthetic spectral data (Beer-Lambert)."""
    k1_true = 1.5
    D_df, S_true = _generate_spectral_data(k1_true=k1_true)

    r = ReactionModel('spectral_pe')
    k1 = r.parameter('k1', value=0.8, bounds=(0.1, 5))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(name='spectral', data=D_df, category='spectral')
    r.set_discretization(nfe=25, ncp=3)

    r.run_opt(variances={'device': 1e-4})
    ro = r.results_dict['p_estimator']

    assert ro.status in ('optimal', 'feasible')
    assert 'k1' in ro.P
    assert abs(ro.P['k1'] - k1_true) / k1_true < 0.1, \
        f"k1 recovered={ro.P['k1']:.4f}, true={k1_true}"


@pytest.mark.slow
def test_non_absorbing_species():
    """Non-absorbing species are excluded from Beer-Lambert."""
    D_df, _ = _generate_spectral_data(k1_true=1.0)

    r = ReactionModel('non_abs')
    k1 = r.parameter('k1', value=0.5, bounds=(0.1, 5))
    A = r.component('A', value=1.0, absorbing=True)
    B = r.component('B', value=0.0, absorbing=False)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.add_data(name='spectral', data=D_df, category='spectral')
    r.set_non_absorbing_species(['B'])
    r.set_discretization(nfe=20, ncp=3)

    # Should build without error even with non-absorbing species
    b = r._make_builder()
    m, dae, dm_params = b.build_parameter_estimation_model({'device': 1.0})
    assert m is not None


def test_spectral_data_loaded():
    """Spectral data is stored as SpectralData object."""
    D_df, _ = _generate_spectral_data()
    r = ReactionModel('spec_load')
    r.add_data(name='spectral', data=D_df, category='spectral')
    assert r.spectra is not None
    assert r.spectra.D is not None
    assert r.spectra.D.shape == D_df.shape
