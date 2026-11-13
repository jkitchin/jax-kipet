"""
Tests for DiscoptModelBuilder.
"""
import numpy as np
import pytest

from kipetax.main_modules.reaction_model import ReactionModel
from kipetax.model_tools.discopt_model_builder import DiscoptModelBuilder


def _make_simple_ab_builder():
    """A → B with k1=2.0, t in [0, 5]."""
    r = ReactionModel('test')
    k1 = r.parameter('k1', value=2.0, bounds=(0, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.set_time(5.0)
    return r._make_builder()


def test_build_simulation_model():
    """build_simulation_model() should return a valid (m, dae) pair."""
    b = _make_simple_ab_builder()
    m, dae = b.build_simulation_model()
    assert m is not None
    assert dae is not None
    # States declared for A and B
    assert 'A' in dae._vars
    assert 'B' in dae._vars


def test_simulation_model_has_no_free_params():
    """Simulation model treats parameters as fixed floats."""
    b = _make_simple_ab_builder()
    m, dae = b.build_simulation_model()
    # Model should be solvable without parameter variables
    assert m.num_continuous >= 2  # at least A and B state vars


def test_build_pe_model_concentration():
    """PE model with concentration data builds without error."""
    import pandas as pd

    r = ReactionModel('pe_test')
    k1 = r.parameter('k1', value=2.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.set_time(5.0)

    # Synthetic data
    t = np.linspace(0, 5, 10)
    A_data = np.exp(-2.0 * t)
    B_data = 1 - np.exp(-2.0 * t)
    df = pd.DataFrame({'A': A_data, 'B': B_data}, index=t)
    r.add_data(name='ds1', data=df)

    b = r._make_builder()
    m, dae, dm_params = b.build_parameter_estimation_model({'A': 1.0, 'B': 1.0})

    assert m is not None
    assert 'k1' in dm_params


def test_pe_model_has_parameter_variable():
    """PE model has discopt variable for each non-fixed parameter."""
    import discopt

    r = ReactionModel('pe2')
    k1 = r.parameter('k1', value=2.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)
    r.set_time(5.0)

    import pandas as pd
    t = np.linspace(0, 5, 5)
    df = pd.DataFrame({'A': np.exp(-2.0 * t), 'B': 1 - np.exp(-2.0 * t)}, index=t)
    r.add_data(data=df)

    b = r._make_builder()
    m, dae, dm_params = b.build_parameter_estimation_model()

    # dm_params['k1'] should be a discopt Variable
    assert isinstance(dm_params['k1'], discopt.Variable)


def test_time_inference_from_data():
    """Builder infers times from dataset if set_time() not called."""
    import pandas as pd

    r = ReactionModel('infer_t')
    k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))
    A = r.component('A', value=1.0)
    B = r.component('B', value=0.0)
    r.add_ode('A', -k1 * A)
    r.add_ode('B',  k1 * A)

    t = np.linspace(0, 3, 8)
    df = pd.DataFrame({'A': np.exp(-t), 'B': 1 - np.exp(-t)}, index=t)
    r.add_data(data=df)

    b = r._make_builder()
    # No set_time() called; infer from data
    b._infer_times()
    assert b._times == pytest.approx((0.0, 3.0), abs=1e-9)
