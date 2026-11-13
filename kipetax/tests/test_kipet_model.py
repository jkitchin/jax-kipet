"""
Tests for the ReactionModel API (ported from kipet test_kipet_model.py).
Covers the public-facing API to ensure compatibility.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax import ReactionModel


class TestReactionModelAPI:
    """Tests for the ReactionModel API."""

    def test_init_default_name(self):
        r = ReactionModel()
        assert r.name == 'Model-1'

    def test_init_custom_name(self):
        r = ReactionModel('test_model')
        assert r.name == 'test_model'

    def test_parameter_returns_proxy(self):
        r = ReactionModel()
        from kipetax.model_tools.proxy_vars import KipetProxy
        k1 = r.parameter('k1', value=2.0, bounds=(0, 10))
        assert isinstance(k1, KipetProxy)
        assert k1.name == 'k1'

    def test_parameter_stored_in_block(self):
        r = ReactionModel()
        r.parameter('k1', value=2.0, bounds=(0, 10))
        assert 'k1' in r.parameters.names
        assert r.parameters['k1'].value == 2.0

    def test_component_returns_proxy(self):
        r = ReactionModel()
        from kipetax.model_tools.proxy_vars import KipetProxy
        A = r.component('A', value=1.0)
        assert isinstance(A, KipetProxy)
        assert A.name == 'A'

    def test_component_stored_in_block(self):
        r = ReactionModel()
        r.component('A', value=1.5, bounds=(0, 5))
        assert 'A' in r.components.names
        assert r.components['A'].value == 1.5

    def test_state_returns_proxy(self):
        r = ReactionModel()
        from kipetax.model_tools.proxy_vars import KipetProxy
        V = r.state('V', value=1.0)
        assert isinstance(V, KipetProxy)

    def test_constant_returns_proxy(self):
        r = ReactionModel()
        from kipetax.model_tools.proxy_vars import KipetProxy
        R = r.constant('R', value=8.314)
        assert isinstance(R, KipetProxy)

    def test_add_ode_stores_expression(self):
        r = ReactionModel()
        k1 = r.parameter('k1', value=1.0, bounds=(0, 5))
        A = r.component('A', value=1.0)
        expr = r.add_ode('A', -k1 * A)
        assert 'A' in r.odes_dict

    def test_add_odes_from_dict(self):
        r = ReactionModel()
        k1 = r.parameter('k1', value=1.0, bounds=(0, 5))
        A = r.component('A', value=1.0)
        B = r.component('B', value=0.0)
        rates = {'A': -k1 * A, 'B': k1 * A}
        r.add_odes(rates)
        assert 'A' in r.odes_dict
        assert 'B' in r.odes_dict

    def test_set_time(self):
        r = ReactionModel()
        r.set_time(10.0)
        assert r._times == (0.0, 10.0)

    def test_set_time_raises_without_arg(self):
        r = ReactionModel()
        with pytest.raises(ValueError):
            r.set_time()

    def test_add_data_dataframe(self):
        r = ReactionModel()
        t = np.linspace(0, 5, 10)
        df = pd.DataFrame({'A': np.exp(-t)}, index=t)
        r.add_data(name='ds1', data=df)
        assert 'ds1' in r.datasets.names

    def test_add_spectral_data(self):
        r = ReactionModel()
        t = np.linspace(0, 5, 5)
        lam = [400, 500, 600]
        D = pd.DataFrame(np.random.rand(5, 3), index=t, columns=lam)
        r.add_data(name='sp1', data=D, category='spectral')
        assert r.spectra is not None

    def test_set_non_absorbing_species(self):
        r = ReactionModel()
        r.component('A', value=1.0)
        r.component('B', value=0.0)
        r.set_non_absorbing_species(['B'])
        assert 'B' in r._non_absorbing

    def test_fix_parameter(self):
        r = ReactionModel()
        r.parameter('k1', value=2.0, bounds=(0, 10))
        r.fix_parameter('k1')
        assert r.parameters['k1'].fixed is True

    def test_set_discretization(self):
        r = ReactionModel()
        r.set_discretization(nfe=50, ncp=4, scheme='legendre')
        assert r._nfe == 50
        assert r._ncp == 4
        assert r._scheme == 'legendre'

    def test_comp_attribute_access(self):
        """comp dict allows attribute-style access to proxy vars."""
        r = ReactionModel()
        k1 = r.parameter('k1', value=2.0, bounds=(0, 5))
        A = r.component('A', value=1.0)
        assert r.comp.k1 is k1
        assert r.comp.A is A

    def test_expression_arithmetic(self):
        """ODE expressions support arithmetic operations."""
        from kipetax.model_tools.proxy_vars import PendingBinaryOp
        r = ReactionModel()
        k1 = r.parameter('k1', value=1.0, bounds=(0, 5))
        A = r.component('A', value=1.0)
        B = r.component('B', value=0.0)
        expr = -k1 * A + 2.0 * B
        assert isinstance(expr, PendingBinaryOp)

    def test_results_dict_keys_after_simulation(self):
        """After simulate(), results_dict has 'simulator' key."""
        r = ReactionModel()
        k1 = r.parameter('k1', value=1.0, bounds=(0, 5), fixed=True)
        A = r.component('A', value=1.0)
        B = r.component('B', value=0.0)
        r.add_ode('A', -k1 * A)
        r.add_ode('B',  k1 * A)
        r.set_time(3.0)
        r.set_discretization(nfe=10, ncp=3)
        r.simulate()
        assert 'simulator' in r.results_dict

    def test_diagnostics_runs(self):
        """diagnostics() prints without errors."""
        r = ReactionModel('diagnostics_test')
        r.parameter('k1', value=1.0, bounds=(0, 5))
        r.component('A', value=1.0)
        r.set_time(5.0)
        r.diagnostics()  # should not raise

    def test_repr(self):
        r = ReactionModel('test')
        assert 'test' in repr(r)

    def test_str(self):
        r = ReactionModel('test')
        r.component('A', value=1.0)
        r.parameter('k1', value=2.0, bounds=(0, 10))
        s = str(r)
        assert 'test' in s


class TestParameterBlock:
    """Tests for ParameterBlock."""

    def test_names(self):
        r = ReactionModel()
        r.parameter('k1', value=1.0, bounds=(0, 5))
        r.parameter('k2', value=0.5, bounds=(0, 2))
        assert set(r.parameters.names) == {'k1', 'k2'}

    def test_values_dict(self):
        r = ReactionModel()
        r.parameter('k1', value=1.5, bounds=(0, 5))
        assert r.parameters['k1'].value == pytest.approx(1.5)

    def test_bounds(self):
        r = ReactionModel()
        r.parameter('k1', value=2.0, bounds=(0.5, 8.0))
        assert r.parameters['k1'].lb == 0.5
        assert r.parameters['k1'].ub == 8.0


class TestComponentBlock:
    """Tests for ComponentBlock."""

    def test_absorbing_default(self):
        r = ReactionModel()
        r.component('A', value=1.0)
        assert r.components['A'].absorbing is True

    def test_absorbing_false(self):
        r = ReactionModel()
        r.component('B', value=0.0, absorbing=False)
        assert r.components['B'].absorbing is False

    def test_initial_value(self):
        r = ReactionModel()
        r.component('A', value=2.5)
        assert r.components['A'].value == pytest.approx(2.5)
