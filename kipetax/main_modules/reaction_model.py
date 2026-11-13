"""
ReactionModel for kipetax (Pyomo-free).

Primary user-facing class. Preserves the kipet ReactionModel API while
replacing all Pyomo internals with discopt via DiscoptModelBuilder.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

from kipetax.model_components.element_blocks import (
    AlgebraicBlock, ComponentBlock, ConstantBlock, ParameterBlock, StateBlock
)
from kipetax.model_components.data_component import DataBlock, DataSet
from kipetax.model_components.spectral_handler import SpectralData
from kipetax.model_tools.discopt_model_builder import DiscoptModelBuilder
from kipetax.estimator_tools.discopt_simulator import DiscoptSimulator
from kipetax.estimator_tools.parameter_estimator import ParameterEstimator
from kipetax.estimator_tools.results_object import ResultsObject


def _nonneg_is_active(spectra_nonneg):
    """Whether the collocation path should keep its hard ``S >= 0`` bound.

    Collocation expresses non-negativity as a variable bound, which has no
    weight, so a numeric ``spectra_nonneg`` collapses to on/off there while
    varpro uses it as a penalty weight.
    """
    if spectra_nonneg is True or spectra_nonneg is False:
        return spectra_nonneg
    return float(spectra_nonneg) > 0


class _AttrDict(dict):
    """A dict that allows attribute-style access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class ReactionModel:
    """
    Primary class for describing and solving kinetic reaction models.

    Users define components, parameters, ODEs (and optionally algebraics
    and spectral data), then call simulate() or run_opt() to solve.

    Example
    -------
    >>> r1 = ReactionModel('my_model')
    >>> k1 = r1.parameter('k1', value=2.0, bounds=(0, 10))
    >>> A = r1.component('A', value=1.0)
    >>> B = r1.component('B', value=0.0)
    >>> r1.add_ode('A', -k1 * A)
    >>> r1.add_ode('B',  k1 * A)
    >>> r1.set_time(10.0)
    >>> r1.simulate()
    """

    def __init__(self, name=None, model=None, designation=None):
        self.name = name if name is not None else 'Model-1'

        # Public model-component containers
        self.components = ComponentBlock()
        self.parameters = ParameterBlock()
        self.constants = ConstantBlock()
        self.algebraics = AlgebraicBlock()
        self.states = StateBlock()
        self.datasets = DataBlock()

        self.results_dict = {}
        self.variances = {}
        self.odes_dict = {}   # {var_name: pending expression}
        self.algs_dict = {}   # {var_name: pending expression}
        self.spectra = None   # SpectralData or None
        self._interactions = {}   # {name: pending expression} absorbing columns
        self._spectral_ridge = 0.0

        self.comp = _AttrDict()  # attribute-access dict of proxy vars

        # Internal settings
        self._nfe = 20
        self._ncp = 3
        self._scheme = 'radau'
        self._element_boundaries = None
        self._times = None            # (t0, tf) set via set_time or data
        self._simulation_time = None  # scalar end time from set_time()
        self._non_absorbing = []
        self._absorbing_components = None
        self._fixed_params = []
        self._optimized = False
        self._is_simulation = False

        # Builder (constructed at solve time)
        self._builder = None

        if model is not None:
            self._copy_from_model(model)

    def __repr__(self):
        return f"ReactionModel({self.name!r})"

    def __str__(self):
        s = f"ReactionModel: {self.name}\n"
        s += str(self.components)
        s += str(self.parameters)
        return s

    # ── Model Component Registration ──

    def parameter(self, name, **kwargs):
        """
        Define a kinetic parameter to be estimated.

        Parameters
        ----------
        name : str
            Parameter name.
        value : float, optional
            Initial / nominal value.
        bounds : tuple, optional
            (lb, ub).
        fixed : bool, optional
            If True, parameter is held constant (not estimated).
        variance : float, optional
            Known variance.

        Returns
        -------
        KipetProxy
            Proxy object usable in ODE expressions.
        """
        self.parameters.add_element(name, **kwargs)
        proxy = self.parameters[name].proxy_var
        self.comp[name] = proxy
        return proxy

    def component(self, name, **kwargs):
        """
        Define a chemical species (concentration-based).

        Parameters
        ----------
        name : str
        value : float
            Initial concentration.
        bounds : tuple, optional
        variance : float, optional
        absorbing : bool, optional
            Default True. Set False for non-absorbing species.

        Returns
        -------
        KipetProxy
        """
        # Initialize ODE to 0 if not set yet (will be overwritten by add_ode)
        if name not in self.odes_dict:
            self.odes_dict[name] = 0

        self.components.add_element(name, **kwargs)
        proxy = self.components[name].proxy_var
        self.comp[name] = proxy
        return proxy

    def state(self, name, **kwargs):
        """
        Define a non-concentration state variable (e.g. volume, temperature).

        Returns
        -------
        KipetProxy
        """
        if name not in self.odes_dict:
            self.odes_dict[name] = 0

        is_volume = kwargs.pop('is_volume', False)

        self.states.add_element(name, **kwargs)
        proxy = self.states[name].proxy_var
        self.comp[name] = proxy
        return proxy

    def volume(self, **kwargs):
        """Convenience method to define volume state named 'V'."""
        return self.state('V', is_volume=True, **kwargs)

    def constant(self, name, **kwargs):
        """
        Define a fixed model constant (not estimated).

        Returns
        -------
        KipetProxy
        """
        self.constants.add_element(name, **kwargs)
        proxy = self.constants[name].proxy_var
        self.comp[name] = proxy
        return proxy

    def algebraic(self, name, **kwargs):
        """
        Define a generic algebraic variable (no time derivative).

        Returns
        -------
        KipetProxy
        """
        self.algebraics.add_element(name, **kwargs)
        proxy = self.algebraics[name].proxy_var
        self.comp[name] = proxy
        return proxy

    def interaction(self, name, expr):
        """
        Declare a non-additive (interaction) contribution to the spectra.

        Beer-Lambert bilinearity assumes the measured spectrum is a
        concentration-weighted sum of fixed pure-component spectra. Real
        systems break that: an A.B adduct has a band whose intensity goes as
        ``c_A * c_B``, and an environment-driven band shift contributes a
        derivative spectrum. Both are captured by adding a column to ``C``
        whose value is an arbitrary expression of the states::

            D(t, nu) = sum_k c_k(t) s_k(nu) + sum_j g_j(t) Gamma_j(nu)

        The fitted ``Gamma_j`` is returned as an extra column of
        ``ResultsObject.S``.

        An interaction is an ALGEBRAIC function of the states, not a
        differential state. Do not give it an ODE via the product rule: that
        adds a spurious quadratic differential constraint to the collocation
        NLP, which is both slower and worse conditioned.

        The augmented model is strictly more flexible than the additive one and
        can trade interaction spectra against the rate constants. Use
        :meth:`set_spectral_regularization` to keep it identified, and decide
        that an interaction term is warranted from rank / structured-residual
        diagnostics before adding it.

        Parameters
        ----------
        name : str
            Label for the interaction spectrum (a column of ``S``).
        expr : pending expression
            Function of components/states, e.g. ``A * B`` or ``C ** 2``.

        Examples
        --------
        >>> AB = r1.interaction('AB', A * B)   # A.B adduct band ~ c_A c_B
        """
        self._interactions[name] = expr
        return expr

    def set_spectral_regularization(self, ridge=0.0):
        """
        Ridge penalty on the interaction spectra (0 disables).

        Adds ``ridge * sum(Gamma**2)`` to the objective. Interaction spectra are
        signed and unconstrained, so without a penalty an augmented model fit to
        data with little or no interaction becomes under-identified.
        """
        self._spectral_ridge = float(ridge)

    def fixed_state(self, name, **kwargs):
        """
        Define an algebraic variable with a fixed trajectory from data.

        The 'data' keyword argument is required.

        Returns
        -------
        KipetProxy
        """
        if 'data' not in kwargs:
            raise ValueError("fixed_state requires 'data' keyword argument")
        return self.algebraic(name, **kwargs)

    def step(self, name, **kwargs):
        """
        Define a step function variable.

        Returns a KipetProxy that can be used in ODE expressions.
        """
        return self.algebraic(name, **kwargs)

    # ── Data ──

    def add_data(self, name=None, file=None, data=None,
                 category=None, remove_negatives=False, time_scale=None):
        """
        Add experimental data to the model.

        Parameters
        ----------
        name : str, optional
        file : str, optional
            Path to CSV file.
        data : DataFrame, optional
        category : str, optional
            'spectral' for D-matrix data.
        remove_negatives : bool, optional
        time_scale : str, optional
            Unused (for API compatibility).
        """
        if name is None:
            name = f'ds-{len(self.datasets) + 1}'

        if file is not None:
            import pandas as pd
            dataframe = pd.read_csv(file, index_col=0)
        elif data is not None:
            dataframe = data
        else:
            raise ValueError("Must provide 'file' or 'data'")

        if remove_negatives:
            dataframe = dataframe.clip(lower=0)

        if category == 'spectral':
            D_data = SpectralData(name, file=file, remove_negatives=remove_negatives)
            D_data.add_data(dataframe)
            self.spectra = D_data
        else:
            dataset = DataSet(name=name, category=category, data=dataframe, file=file)
            self.datasets.add_dataset(dataset)

    # ── ODE/Algebraic Expressions ──

    def add_ode(self, ode_var, expr):
        """
        Set the ODE for a component or state.

        Parameters
        ----------
        ode_var : str
            Name of the component or state variable.
        expr : pending expression or numeric
            RHS of the ODE (dx/dt = expr).

        Returns
        -------
        The expression (for chaining / display).
        """
        self.odes_dict[ode_var] = expr
        return expr

    def add_odes(self, ode_dict):
        """
        Add multiple ODEs from a dictionary.

        Parameters
        ----------
        ode_dict : dict
            {var_name: expression}
        """
        if isinstance(ode_dict, dict):
            for key, val in ode_dict.items():
                self.add_ode(key, val)

    def add_expression(self, name, expr, **kwargs):
        """
        Add a named algebraic expression.

        Parameters
        ----------
        name : str
        expr : pending expression

        Returns
        -------
        The expression.
        """
        is_reaction = kwargs.pop('is_reaction', False)
        active = kwargs.pop('active', True)
        bypass = kwargs.pop('bypass', True)

        if not bypass:
            if name not in self.algebraics.names:
                self.algebraics.add_element(name, is_reaction=is_reaction,
                                            active=active, **kwargs)
                proxy = self.algebraics[name].proxy_var
                self.comp[name] = proxy

            if not active:
                expr = 0

            self.algs_dict[name] = expr

        return expr

    def add_reaction(self, name, expr, **kwargs):
        """
        Declare a named reaction expression.

        Returns the expression (usable in ODE RHS).
        """
        kwargs['is_reaction'] = True
        kwargs['bypass'] = False
        return self.add_expression(name, expr, **kwargs)

    def add_constraint(self, expr):
        """Add a custom algebraic constraint (NLP constraint)."""
        # For now, store as algebraic; full support in future
        return expr

    def add_objective_from_algebraic(self, algebraic_var):
        """Declare a custom objective using an algebraic variable (kipet API compat)."""
        self._custom_objective = algebraic_var

    # ── Time ──

    def set_time(self, time_span=None):
        """
        Set the simulation/estimation end time.

        Parameters
        ----------
        time_span : float
            Duration (t0=0 assumed).
        """
        if time_span is None:
            raise ValueError("Must provide a time span")
        self._simulation_time = float(time_span)
        self._times = (0.0, float(time_span))

    # ── Discretization settings ──

    def set_discretization(self, nfe=20, ncp=3, scheme='radau',
                           element_boundaries=None):
        """
        Configure the collocation discretization.

        Parameters
        ----------
        nfe : int
            Number of finite elements. Ignored when ``element_boundaries`` is
            given (it is then inferred as ``len(element_boundaries) - 1``).
        ncp : int
            Collocation points per element.
        scheme : str
            'radau' or 'legendre'.
        element_boundaries : array-like, optional
            Explicit, strictly increasing element boundaries -- a GRADED mesh.

            A uniform mesh silently fails on a system with a fast initial
            transient. A free-radical moment model whose radicals relax to
            quasi-steady state in ~0.05 time units, observed over 10, converges
            to a wrong rate constant with `status='optimal'` at nfe=40, 80 and
            160 alike, because refining uniformly never resolves the layer.
            Cluster elements where the solution moves::

                eb = np.concatenate([np.linspace(0, 0.5, 20),
                                     np.linspace(0.5, 10.0, 21)[1:]])
                r.set_discretization(ncp=3, element_boundaries=eb)
        """
        self._nfe = nfe
        self._ncp = ncp
        self._scheme = scheme
        self._element_boundaries = element_boundaries

    # ── Non-absorbing / absorbing species ──

    def set_non_absorbing_species(self, species_list):
        """Mark species as non-absorbing (excluded from Beer-Lambert)."""
        self._non_absorbing = list(species_list)

    def set_known_absorbing_species(self, species_list):
        """Explicitly set which species appear in the Beer-Lambert relation."""
        self._absorbing_components = list(species_list)

    # ── Parameter manipulation ──

    def fix_parameter(self, param_name):
        """Fix a parameter at its current value (do not estimate)."""
        if param_name in self.parameters.names:
            self.parameters[param_name].fixed = True
            self._fixed_params.append(param_name)

    def bound_profile(self, var, bounds):
        """Set bounds on a component or state (API compatibility)."""
        for block in [self.components, self.states, self.algebraics]:
            if var in block.names:
                block[var].bounds = bounds
                return

    def initialize_from_trajectory(self, var_name, data):
        """Initialize a variable from trajectory data (API compatibility)."""
        pass  # Future: warm-start from data

    # ── Builder construction ──

    def _make_builder(self):
        """Construct and populate a DiscoptModelBuilder from this ReactionModel."""
        b = DiscoptModelBuilder(name=self.name)

        # Transfer elements
        for comp in self.components:
            b.add_component(comp)
        for param in self.parameters:
            b.add_parameter(param)
        for state in self.states:
            b.add_state(state)
        for alg in self.algebraics:
            b.add_algebraic(alg)
        for const in self.constants:
            b.add_constant(const)

        # Transfer ODE expressions (raw pending trees)
        b.set_odes(self.odes_dict)
        b.set_algebraics(self.algs_dict)

        # Transfer data
        b.set_datasets(self.datasets)
        b.set_spectral_data(self.spectra)
        b.set_interactions(self._interactions)
        b.set_spectral_ridge(self._spectral_ridge)

        # Set non-absorbing / absorbing
        b._non_absorbing = self._non_absorbing
        b._absorbing_components = self._absorbing_components

        # Set times
        if self._times is not None:
            b.set_times(*self._times)

        # Set discretization
        b.set_discretization(nfe=self._nfe, ncp=self._ncp, scheme=self._scheme,
                             element_boundaries=self._element_boundaries)

        self._builder = b
        return b

    # ── Simulation ──

    def simulate(self, parameters=None, solver_opts=None):
        """
        Run a dynamic simulation.

        Parameters
        ----------
        parameters : dict, optional
            Override parameter values for this run.
        solver_opts : dict, optional
            Options forwarded to POUNCE, e.g. ``{'max_iter': 200}``. A
            simulation is a pure feasibility solve with no objective to guide
            it, so a stiff model can iterate for a very long time. Cap it and
            handle the resulting ``RuntimeError`` rather than waiting.

        Returns
        -------
        None (results stored in results_dict['simulator']).

        Raises
        ------
        RuntimeError
            If the collocation solve does not converge. The trajectory would
            otherwise be empty.
        """
        if len(self.odes_dict) == 0:
            raise ValueError("No ODEs defined")

        if self._times is None:
            raise ValueError("Call set_time() before simulating")

        b = self._make_builder()
        sim = DiscoptSimulator(b)
        ro = sim.run_sim(parameters=parameters, solver_options=solver_opts)
        self.results_dict['simulator'] = ro
        self._is_simulation = True

        if ro.status not in ('optimal', 'feasible'):
            # Fail loudly. A failed collocation solve leaves Z/X empty, and
            # silently handing back empty DataFrames makes the caller debug a
            # KeyError far from the cause. Stiff systems land here.
            raise RuntimeError(
                f"Simulation failed: solver status={ro.status}. The collocation "
                f"solve did not converge, so no trajectory is available. Try "
                f"more finite elements (set_discretization(nfe=...)), or bound "
                f"the states; a stiff system with widely separated magnitudes "
                f"may need a graded mesh.")

        print(f"Simulation complete: status={ro.status}")
        return None

    # ── Parameter Estimation ──

    def run_opt(self, variances=None, covariance=False, method='collocation',
                spectra_nonneg=True, **kwargs):
        """
        Run parameter estimation.

        Parameters
        ----------
        variances : dict, optional
            Component variances. If not given, uses sigma_sq from self.variances.
        covariance : bool, optional
            If True, compute confidence intervals.
        spectra_nonneg : bool or float, optional
            Whether the recovered SPECIES spectra are pushed non-negative.
            ``True`` (default) is a hard ``S >= 0`` bound under
            ``'collocation'`` and a penalty of weight 2.0 under ``'varpro'``;
            ``False`` removes it from either. A float sets the varpro penalty
            weight directly and behaves as ``True``/``False`` under collocation
            according to whether it is positive.

            Set it to ``False`` (or a small weight) **when the D-matrix has been
            baseline-corrected or scatter-corrected.** Non-negativity is a
            statement about raw absorbance. MSC applies a per-spectrum affine
            map -- subtract an intercept, divide by a slope -- which is not
            Beer-Lambert-preserving, so the pure-component spectra of
            preprocessed data have genuinely negative regions; a minor component
            can easily carry ~10% of its band energy there. Imposing ``S >= 0``
            anyway is a false constraint, and it biases the *kinetics*, not just
            the spectra, because the fit trades rate constants against the
            forbidden negative lobes.

            Concentration bounds are unaffected: ``C >= 0`` holds regardless of
            what was done to the spectra.
        method : {'collocation', 'varpro'}, optional
            Solver strategy.

            * ``'collocation'`` (default) -- the SIMULTANEOUS problem: the ODEs
              are transcribed by orthogonal collocation and solved with POUNCE,
              with the pure-component spectra ``S`` as decision variables. Gives
              joint confidence intervals over ``S`` and the kinetics, and hard
              constraints on ``S``, but its cost grows quadratically in the
              number of wavelengths.
            * ``'varpro'`` -- variable projection: ``S`` is eliminated
              analytically (``S* = C^+ D``) so the optimizer sees only the
              kinetic parameters, making the cost independent of the wavelength
              count. Integrates the ODEs by sequential shooting (differentiable
              RK4). This is the right tool for a wide spectral axis. Only
              available for spectral fits.

        Returns
        -------
        None (results stored in results_dict['p_estimator']).
        """
        if not self.odes_dict:
            raise ValueError("No ODEs defined")

        # Determine times from data if not set
        if self._times is None:
            self._infer_times_from_data()
        if self._times is None:
            raise ValueError("Cannot determine time range; call set_time()")

        # Merge variances
        all_variances = dict(self.variances)
        if variances is not None:
            all_variances.update(variances)

        b = self._make_builder()
        b._spectra_nonneg = _nonneg_is_active(spectra_nonneg)
        solver_opts = kwargs.pop('solver_opts', {})
        cov_method = kwargs.pop('cov_method', 'analytic')

        if method == 'varpro':
            if b._spectral_data is None or b._spectral_data.D is None:
                raise ValueError("method='varpro' requires spectral data "
                                 "(add_data(..., category='spectral'))")
            from kipetax.estimator_tools.varpro_estimator import VarProEstimator
            ro = VarProEstimator(b).run_opt(variances=all_variances,
                                            covariance=covariance,
                                            spectra_nonneg=spectra_nonneg)
        elif method == 'collocation':
            pe = ParameterEstimator(b)
            ro = pe.run_opt(variances=all_variances, solver_options=solver_opts,
                            covariance=covariance, cov_method=cov_method)
        else:
            raise ValueError(f"unknown method {method!r}; "
                             f"use 'collocation' or 'varpro'")

        self.results_dict['p_estimator'] = ro
        self._optimized = True

        if ro.status not in ('optimal', 'feasible'):
            hint = ""
            if 'UNBOUNDED' in str(ro.status):
                # A weighted sum of squares is bounded below by zero, so this is
                # never a genuinely unbounded problem -- it is objective scaling.
                hint = (" A least-squares objective cannot be unbounded below; "
                        "this is a scaling failure. The residuals are weighted "
                        "by 1/variance, so a very small variance makes the "
                        "objective enormous. Supply a realistic measurement "
                        "variance (1/variance beyond ~1e7 is not reliable), or "
                        "rescale the states.")
            raise RuntimeError(
                f"Parameter estimation failed: solver status={ro.status}. No "
                f"parameter estimates are available.{hint} Partial results are "
                f"in results_dict['p_estimator'].")

        print(f"Parameter estimation complete: status={ro.status}")
        if ro.P:
            for k, v in ro.P.items():
                print(f"  {k:20s} = {v:.6g}")

        return None

    def _infer_times_from_data(self):
        """Determine (t0, tf) from the loaded datasets and spectra."""
        t_all = []
        for ds in self.datasets:
            if ds.data is not None:
                t_all.extend(ds.data.index.tolist())
        if self.spectra is not None and self.spectra.D is not None:
            t_all.extend(self.spectra.times)
        if t_all:
            from kipetax.model_tools.discopt_model_builder import _initial_time
            self._times = (_initial_time(float(min(t_all))), float(max(t_all)))

    # ── Variance estimation ──

    def run_variance_estimator(self, **kwargs):
        """
        Run variance estimation (wraps VarianceEstimator).

        Stores variances in self.variances.
        """
        from kipetax.estimator_tools.variance_estimator import VarianceEstimator

        if self._times is None:
            self._infer_times_from_data()

        b = self._make_builder()
        ve = VarianceEstimator(b)
        variances = ve.run(**kwargs)
        self.variances.update(variances)
        self.results_dict['v_estimator'] = variances
        return variances

    # ── Plotting ──

    def plot(self, var='Z', filename=None, show=True):
        """
        Plot simulation or estimation results.

        Parameters
        ----------
        var : str
            'Z' (concentrations), 'X' (states), or 'Y' (algebraics).
        filename : str, optional
            Save to file.
        show : bool
            Display plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; cannot plot")
            return

        # Find the latest results
        ro = None
        for key in ('p_estimator', 'v_estimator', 'simulator'):
            if key in self.results_dict:
                ro = self.results_dict[key]
                break

        if ro is None or not isinstance(ro, ResultsObject):
            print("No results available to plot")
            return

        df = getattr(ro, var, None)
        if df is None or (hasattr(df, 'empty') and df.empty):
            print(f"No data for variable '{var}'")
            return

        fig, ax = plt.subplots()
        for col in df.columns:
            ax.plot(df.index, df[col], label=col)
        ax.set_xlabel('Time')
        ax.set_ylabel(var)
        ax.legend()
        ax.set_title(f'{self.name} - {var}')

        if filename:
            fig.savefig(filename)
        if show:
            plt.show()

    # ── API compatibility helpers ──

    def _copy_from_model(self, model):
        """Copy components from another ReactionModel instance."""
        if not isinstance(model, ReactionModel):
            raise ValueError("Can only copy from ReactionModel instances")
        for attr in ['components', 'parameters', 'constants', 'algebraics',
                     'states', 'odes_dict', 'algs_dict']:
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

    def make_model(self):
        """Build the discopt model (for compatibility with kipet API)."""
        self._make_builder()
        return self._builder

    @property
    def all_params(self):
        """Return dict of all parameter name -> value."""
        return self.parameters.values

    def diagnostics(self):
        """Print a summary of the model."""
        print(f"ReactionModel: {self.name}")
        print(f"  Components : {self.components.names}")
        print(f"  Parameters : {self.parameters.names}")
        print(f"  States     : {self.states.names}")
        print(f"  Algebraics : {self.algebraics.names}")
        print(f"  Constants  : {self.constants.names}")
        print(f"  ODEs       : {list(self.odes_dict.keys())}")
        print(f"  Times      : {self._times}")
        print(f"  NFE/NCP    : {self._nfe}/{self._ncp}")
        print(f"  Datasets   : {self.datasets.names}")
        print(f"  Spectra    : {'yes' if self.spectra is not None else 'no'}")
