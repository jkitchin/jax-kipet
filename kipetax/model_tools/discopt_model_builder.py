"""
DiscoptModelBuilder for kipetax.

Replaces TemplateBuilder. Accumulates model spec and builds discopt models
for simulation and parameter estimation.
"""
import warnings

import numpy as np
import pandas as pd

from kipetax.model_tools.proxy_vars import resolve_expression


def _balanced_sum(terms):
    """Sum expression terms pairwise, so the result tree has depth O(log n).

    A least-squares objective has one term per measurement. Folding them left
    (``total = total + t``) builds a chain as deep as the dataset is long, and
    discopt walks expression trees recursively, so ~1000 measurements exceeded
    Python's recursion limit. Pairwise summation keeps the depth at
    ``ceil(log2 n)`` (11 for 2000 terms) and adds the same terms.
    """
    terms = list(terms)
    if not terms:
        raise ValueError("cannot sum an empty list of terms")
    while len(terms) > 1:
        paired = [terms[i] + terms[i + 1] for i in range(0, len(terms) - 1, 2)]
        if len(terms) % 2:
            paired.append(terms[-1])
        terms = paired
    return terms[0]


def _initial_time(t_min):
    """The collocation domain must start where the initial conditions hold.

    A component's ``value=`` is its concentration at ``t0``, and
    ``set_time(span)`` fixes ``t0 = 0``. Inferring ``t0`` from the data instead
    silently imposes the initial conditions at the FIRST MEASUREMENT. When the
    data starts late that is a different model, and it fails quietly: on a
    free-radical moment system with data from t=0.5, it returns ``kp = 1375``
    with ``status='optimal'`` where the true value is 1000.

    So integrate from 0 whenever the data starts later, and say so.
    """
    if t_min > 0.0:
        warnings.warn(
            f"Measurements start at t={t_min:g}, but initial conditions are "
            f"declared at t=0. Integrating from t=0; call set_time() to set the "
            f"horizon explicitly, or shift your data if the initial conditions "
            f"really hold at t={t_min:g}.",
            UserWarning, stacklevel=3)
        return 0.0
    return t_min


def solve_local_nlp(m, parameter_values=None, solver_options=None, evaluator=None,
                    initial_guess=None):
    """Solve any discopt model with a single local NLP solve (POUNCE).

    This is the single choke point through which *every* kipetax solve is
    routed, so discopt always uses POUNCE rather than falling back to its
    spatial branch-and-bound. discopt's default ``m.solve()`` auto-detects
    nonconvexity and switches to branch-and-bound -- correct but intractable
    for collocation-sized least-squares -- so it must never be called directly
    in the kipetax pipeline.

    Parameters
    ----------
    m : discopt.Model
        A model with declared variables, constraints, and objective.
    parameter_values : dict, optional
        ``{variable_name: value}`` used to seed ``x0`` at the user's initial
        guesses (a far better start than the bound midpoint). Keys that do not
        match a variable, and ``None`` values, are ignored.
    solver_options : dict, optional
        Options forwarded to ``solve_nlp`` (e.g. ``max_iter``, ``tol``).
    evaluator : discopt NLPEvaluator, optional
        A pre-built evaluator for ``m`` to reuse instead of compiling a fresh
        one. Pass this to re-solve the same model many times (e.g. across
        finite-difference parameter perturbations) without a new JAX compile;
        the evaluator's XLA cache keys on shapes only, so mutating the model's
        ``Parameter`` values between calls hits the cache.
    initial_guess : dict, optional
        ``{variable_name: array}`` warm-start values for full variable blocks
        (e.g. the spectral ``S_matrix``). A bilinear spectral fit started from
        the bound midpoint converges to a poor local minimum; seeding ``S`` from
        a forward simulation (``S = pinv(C) @ D``) lands it at the good optimum.

    Returns
    -------
    discopt.modeling.core.SolveResult
    """
    from kipetax.model_tools.discopt_compat import get_nlp_evaluator
    NLPEvaluator = get_nlp_evaluator()
    from discopt.solvers import SolveStatus
    from discopt.solvers.nlp_pounce import solve_nlp
    from discopt.modeling.core import SolveResult

    ev = evaluator if evaluator is not None else NLPEvaluator(m)
    lb, ub = ev.variable_bounds

    # Start at the midpoint of finite bounds, then seed any supplied
    # variable values over the top.
    x0 = 0.5 * (np.clip(lb, -1e2, 1e2) + np.clip(ub, -1e2, 1e2))
    offset = 0
    var_offsets = {}
    var_sizes = {}
    for v in m._variables:
        var_offsets[v.name] = offset
        var_sizes[v.name] = v.size
        offset += v.size
    if parameter_values:
        for vname, val in parameter_values.items():
            if vname in var_offsets and val is not None:
                x0[var_offsets[vname]] = float(val)
    if initial_guess:
        for vname, val in initial_guess.items():
            base = var_offsets.get(vname)
            if base is None or val is None:
                continue
            arr = np.ravel(np.asarray(val, dtype=np.float64))
            n = min(arr.size, var_sizes[vname])
            x0[base:base + n] = arr[:n]

    nlp_result = solve_nlp(ev, x0, options=solver_options or {})

    x_dict = {}
    o = 0
    for v in m._variables:
        chunk = np.asarray(nlp_result.x[o:o + v.size], dtype=np.float64)
        x_dict[v.name] = chunk.reshape(v.shape) if v.shape else chunk.reshape(())
        o += v.size

    status = ('optimal' if nlp_result.status == SolveStatus.OPTIMAL
              else str(nlp_result.status))
    return SolveResult(status=status, objective=float(nlp_result.objective),
                       x=x_dict)


class DiscoptModelBuilder:
    """
    Builds discopt models from a ReactionModel specification.

    Responsibilities:
    - Accumulate components, parameters, ODEs, algebraics, data
    - Build a discopt Model + DAEBuilder and discretize at solve time
    - Add measurement residuals as NLS objective
    - Add Beer-Lambert constraints for spectral data
    """

    def __init__(self, name='kipetax_model'):
        self.name = name
        self._components = {}    # name -> ModelComponent
        self._parameters = {}    # name -> ModelParameter
        self._states = {}        # name -> ModelState
        self._algebraics = {}    # name -> ModelAlgebraic
        self._constants = {}     # name -> ModelConstant

        self._odes_dict = {}     # name -> pending expression
        self._algs_dict = {}     # name -> pending expression
        self._interactions = {}  # name -> pending expression (absorbing column)
        self._spectral_ridge = 0.0   # ridge weight on the interaction spectra
        # Whether the SPECIES spectra are constrained non-negative. True is
        # correct for raw absorbance; see ``spectra_nonneg`` in
        # ``ReactionModel.run_opt`` for why baseline- or MSC-corrected data
        # needs it relaxed.
        self._spectra_nonneg = True

        self._datasets = None    # DataBlock
        self._spectral_data = None  # SpectralData or None

        # Discretization settings
        self._nfe = 20
        self._ncp = 3
        self._times = (0, 1)
        self._scheme = 'radau'
        self._element_boundaries = None   # graded mesh, or None for uniform

        self._non_absorbing = []   # components not to include in BLL
        self._absorbing_components = None  # set at build time

    # ── Spec accumulation ──

    def add_component(self, comp):
        self._components[comp.name] = comp

    def add_parameter(self, param):
        self._parameters[param.name] = param

    def add_state(self, state):
        self._states[state.name] = state

    def add_algebraic(self, alg):
        self._algebraics[alg.name] = alg

    def add_constant(self, const):
        self._constants[const.name] = const

    def set_odes(self, odes_dict):
        """odes_dict: {var_name: pending_expression}"""
        self._odes_dict = odes_dict

    def set_algebraics(self, algs_dict):
        """algs_dict: {var_name: pending_expression}"""
        self._algs_dict = algs_dict

    def set_datasets(self, datasets):
        self._datasets = datasets

    def set_spectral_data(self, spectral_data):
        self._spectral_data = spectral_data

    def set_interactions(self, interactions):
        """interactions: {name: pending_expression} -> extra absorbing columns."""
        self._interactions = dict(interactions or {})

    def set_spectral_ridge(self, ridge):
        """Ridge weight applied to the interaction spectra (0 disables)."""
        self._spectral_ridge = float(ridge or 0.0)

    def _absorbing_names(self):
        """Absorbing *species* columns of C, in a stable order."""
        if self._absorbing_components is not None:
            return list(self._absorbing_components)
        return [n for n in self._components
                if self._components[n].absorbing and n not in self._non_absorbing]

    def _interaction_columns_numeric(self, Z, X=None):
        """Evaluate each interaction expression on simulated trajectories.

        Used only to warm-start ``C_interaction``; ``resolve_expression`` works
        on numpy arrays exactly as it does on discopt expressions.
        """
        subst = {}
        for col in Z.columns:
            subst[col] = Z[col].values
        if X is not None:
            for col in X.columns:
                subst[col] = X[col].values
        for name, p in self._parameters.items():
            subst[name] = p.value if p.value is not None else 0.0
        for name, c in self._constants.items():
            subst[name] = c.value if c.value is not None else 0.0

        cols = {}
        for name, expr in self._interactions.items():
            cols[name] = np.asarray(resolve_expression(expr, subst), dtype=float)
        return cols

    def set_times(self, start, end):
        self._times = (start, end)

    def set_discretization(self, nfe=20, ncp=3, scheme='radau',
                           element_boundaries=None):
        self._element_boundaries = (None if element_boundaries is None
                                    else np.asarray(element_boundaries, dtype=float))
        if self._element_boundaries is not None:
            nfe = len(self._element_boundaries) - 1
        self._nfe = nfe
        self._ncp = ncp
        self._scheme = scheme

    def _continuous_set(self):
        """Build the collocation domain, honouring a graded mesh if supplied."""
        from discopt.dae.collocation import ContinuousSet
        if self._element_boundaries is not None:
            return ContinuousSet('t', bounds=self._times, ncp=self._ncp,
                                 scheme=self._scheme, nfe=self._nfe,
                                 element_boundaries=self._element_boundaries)
        return ContinuousSet('t', bounds=self._times, nfe=self._nfe,
                             ncp=self._ncp, scheme=self._scheme)

    # ── Time inference ──

    def _infer_times(self):
        """Determine time bounds from datasets and spectral data."""
        t_min, t_max = None, None

        if self._datasets is not None:
            for ds in self._datasets:
                if ds.data is not None and len(ds.data) > 0:
                    t = ds.data.index
                    t_min = min(t_min, t.min()) if t_min is not None else t.min()
                    t_max = max(t_max, t.max()) if t_max is not None else t.max()

        if self._spectral_data is not None and self._spectral_data.D is not None:
            t = self._spectral_data.times
            if t:
                t_min = min(t_min, min(t)) if t_min is not None else min(t)
                t_max = max(t_max, max(t)) if t_max is not None else max(t)

        if t_min is not None and t_max is not None:
            self._times = (_initial_time(float(t_min)), float(t_max))

    # ── Concentration data collection ──

    def _get_concentration_data(self):
        """
        Collect measured concentration data from datasets.

        Returns dict: {component_name: DataFrame with time index and single column}
        """
        c_data = {}
        if self._datasets is None:
            return c_data

        all_comp_names = set(self._components.keys()) | set(self._states.keys())

        for ds in self._datasets:
            if ds.data is None:
                continue
            for col in ds.data.columns:
                if col in all_comp_names:
                    if col not in c_data:
                        c_data[col] = ds.data[[col]].dropna()
                    else:
                        c_data[col] = pd.concat([c_data[col], ds.data[[col]].dropna()])

        return c_data

    # ── Model building helpers ──

    def _make_subst_base(self, dm_params, is_simulation=False):
        """
        Build the base substitution dict (parameters + constants).
        For simulation: parameters are fixed to their values (floats).
        For estimation: parameters are discopt Variable elements.
        """
        subst_base = {}

        if is_simulation:
            for name, p in self._parameters.items():
                subst_base[name] = p.value if p.value is not None else 0.0
        else:
            subst_base.update(dm_params)

        for name, c in self._constants.items():
            subst_base[name] = c.value if c.value is not None else 0.0

        return subst_base

    def _get_bounds(self, elem, default_lb=0.0, default_ub=None):
        """Get safe lb/ub for a model element."""
        lb = elem.bounds[0] if elem.bounds[0] is not None else default_lb
        ub = elem.bounds[1] if elem.bounds[1] is not None else (1e20 if default_ub is None else default_ub)
        return float(lb), float(ub)

    def _get_param_bounds(self, param):
        """Get parameter bounds with fallback."""
        lb = param.bounds[0] if param.bounds[0] is not None else 0.0
        ub = param.bounds[1] if param.bounds[1] is not None else 1e10
        return float(lb), float(ub)

    def _build_ode_rhs(self, subst_base):
        """
        Return a callable (t, states, algebraics, controls) -> dict
        suitable for dae.set_ode().
        """
        odes_dict = self._odes_dict

        def ode_rhs(t, states, algebraics, controls):
            subst = {**subst_base, **states, **algebraics}
            result = {}
            for var_name, expr in odes_dict.items():
                result[var_name] = resolve_expression(expr, subst)
            return result

        return ode_rhs

    def _build_alg_rhs(self, subst_base):
        """
        Return a callable (t, states, algebraics, controls) -> dict
        suitable for dae.set_algebraic().
        """
        algs_dict = self._algs_dict

        def alg_rhs(t, states, algebraics, controls):
            subst = {**subst_base, **states, **algebraics}
            result = {}
            for var_name, expr in algs_dict.items():
                result[var_name] = resolve_expression(expr, subst)
            return result

        return alg_rhs

    def _add_all_states_to_dae(self, dae):
        """Declare all state variables in the DAE."""
        for name, comp in self._components.items():
            lb, ub = self._get_bounds(comp, default_lb=0.0)
            iv = comp.value if comp.value is not None else 0.0
            dae.add_state(name, bounds=(lb, ub), initial=iv)

        for name, state in self._states.items():
            lb, ub = self._get_bounds(state, default_lb=-1e20)
            iv = state.value if state.value is not None else 0.0
            dae.add_state(name, bounds=(lb, ub), initial=iv)

        for name, alg in self._algebraics.items():
            lb, ub = self._get_bounds(alg, default_lb=-1e20)
            dae.add_algebraic(name, bounds=(lb, ub))

    # ── Public build methods ──

    def build_simulation_model(self):
        """
        Build a discopt Model for pure simulation (all parameters fixed to their values).

        Returns
        -------
        (m, dae) : (discopt.Model, DAEBuilder)
        """
        import discopt.modeling as dm
        from discopt.dae.collocation import DAEBuilder, ContinuousSet

        if self._times == (0, 1) and self._datasets is not None:
            self._infer_times()

        m = dm.Model(self.name + '_sim')
        cs = self._continuous_set()
        dae = DAEBuilder(m, cs)

        self._add_all_states_to_dae(dae)

        subst_base = self._make_subst_base({}, is_simulation=True)
        dae.set_ode(self._build_ode_rhs(subst_base))
        if self._algs_dict:
            dae.set_algebraic(self._build_alg_rhs(subst_base))

        dae.discretize()

        # Trivial objective: minimize 0 (feasibility solve)
        dummy = m.continuous('_dummy_obj', lb=0, ub=0)
        m.minimize(dummy)

        return m, dae

    def build_parametric_simulation_model(self):
        """Simulation model with kinetic parameters as discopt ``Parameter`` objects.

        Identical to :meth:`build_simulation_model` except the kinetic
        parameters enter the ODE as discopt ``Parameter`` objects rather than
        baked-in float constants. The compiled evaluator keys its XLA cache on
        shapes only, so the resulting model can be re-solved at many parameter
        values by mutating ``Parameter.value`` -- **without** a fresh JAX
        compile each time. This is what makes the finite-difference covariance
        (one re-solve per perturbed parameter) cheap.

        Returns
        -------
        (m, dae, param_map) : (discopt.Model, DAEBuilder, dict)
            ``param_map`` maps each kinetic parameter name to its discopt
            ``Parameter`` so callers can set ``param_map[name].value`` between
            solves.
        """
        import discopt.modeling as dm
        from discopt.dae.collocation import DAEBuilder, ContinuousSet

        if self._times == (0, 1) and (self._datasets is not None
                                      or self._spectral_data is not None):
            self._infer_times()

        m = dm.Model(self.name + '_simp')
        cs = self._continuous_set()
        dae = DAEBuilder(m, cs)

        self._add_all_states_to_dae(dae)

        # Kinetic parameters as discopt Parameters (fn(x, params) plumbing).
        param_map = {}
        subst_base = {}
        for name, p in self._parameters.items():
            prm = m.parameter(name, float(p.value) if p.value is not None else 0.0)
            param_map[name] = prm
            subst_base[name] = prm
        for name, c in self._constants.items():
            subst_base[name] = c.value if c.value is not None else 0.0

        dae.set_ode(self._build_ode_rhs(subst_base))
        if self._algs_dict:
            dae.set_algebraic(self._build_alg_rhs(subst_base))

        dae.discretize()

        dummy = m.continuous('_dummy_obj', lb=0, ub=0)
        m.minimize(dummy)

        return m, dae, param_map

    def build_parameter_estimation_model(self, variances=None):
        """
        Build a discopt Model for parameter estimation with NLS objective.

        Parameters
        ----------
        variances : dict or None
            Component variances {name: sigma_sq}. If None, uses 1.0 for all.

        Returns
        -------
        (m, dae, dm_params) : (discopt.Model, DAEBuilder, dict)
            dm_params maps parameter names to discopt Variable elements.
        """
        import discopt.modeling as dm
        from discopt.dae.collocation import DAEBuilder, ContinuousSet

        if self._times == (0, 1) and (self._datasets is not None or self._spectral_data is not None):
            self._infer_times()

        if variances is None:
            variances = {}

        m = dm.Model(self.name + '_pe')
        cs = self._continuous_set()
        dae = DAEBuilder(m, cs)

        # Declare state variables
        self._add_all_states_to_dae(dae)

        # Declare parameters as discopt continuous variables
        dm_params = {}
        for name, p in self._parameters.items():
            if p.fixed:
                # Fixed parameters: use value as constant
                dm_params[name] = p.value if p.value is not None else 0.0
            else:
                lb, ub = self._get_param_bounds(p)
                var = m.continuous(f'P_{name}', lb=lb, ub=ub)
                if p.value is not None:
                    # Set initial value via fixing bounds temporarily - discopt sets initial via value attr
                    pass
                dm_params[name] = var

        subst_base = self._make_subst_base(dm_params, is_simulation=False)
        dae.set_ode(self._build_ode_rhs(subst_base))
        if self._algs_dict:
            dae.set_algebraic(self._build_alg_rhs(subst_base))

        dae.discretize()

        # Build objective.
        #
        # Spectral and concentration terms are ADDITIVE, not alternatives. A
        # spectral fit leaves a scale ambiguity (c -> a*c, S -> S/a is a perfect
        # trade), so a handful of direct concentration measurements -- an offline
        # GC endpoint, a calibrated product channel -- is often what pins the
        # absolute scale. Both terms are already weighted by 1/variance, so they
        # are commensurate and simply add.
        obj_terms = []

        if self._spectral_data is not None and self._spectral_data.D is not None:
            obj_terms += self._build_spectral_objective(m, dae, dm_params, variances)

        obj_terms += self._build_concentration_objective(m, dae, variances)

        if obj_terms:
            m.minimize(_balanced_sum(obj_terms))
        else:
            dummy = m.continuous('_dummy_obj', lb=0, ub=0)
            m.minimize(dummy)

        return m, dae, dm_params

    def solve_local(self, m, solver_options=None, initial_guess=None, evaluator=None):
        """Solve a model with a single local NLP solve (POUNCE interior point).

        The parameter-estimation objective is a nonconvex weighted least-squares,
        for which discopt's default ``m.solve()`` falls back to spatial
        branch-and-bound -- correct but intractable for collocation-sized
        problems. Parameter estimation only needs a local optimum, so this
        routes through ``discopt.solvers.nlp_pounce.solve_nlp`` and wraps the
        flat solution into a discopt :class:`SolveResult` so the rest of the
        pipeline (``extract_results``, analytic covariance) is unchanged.

        Parameters
        ----------
        m : discopt.Model
            A model with declared variables, constraints, and objective.
        solver_options : dict, optional
            Options forwarded to ``solve_nlp`` (e.g. ``max_iter``, ``tol``).
        initial_guess : dict, optional
            ``{variable_name: array}`` warm-start values forwarded to
            :func:`solve_local_nlp` (e.g. a seeded spectral ``S_matrix``).
            Blocks not supplied fall back to the constant-trajectory seed from
            :meth:`constant_state_guess`.

        Returns
        -------
        discopt.modeling.core.SolveResult
        """
        # Seed each free parameter at its user-supplied initial guess.
        param_seed = {f'P_{name}': p.value
                      for name, p in self._parameters.items()
                      if p.value is not None}

        # Never let a collocation state start at the bound midpoint.
        guess = self.constant_state_guess()
        if initial_guess:
            guess.update(initial_guess)

        return solve_local_nlp(m, parameter_values=param_seed,
                               solver_options=solver_options,
                               initial_guess=guess, evaluator=evaluator)

    def constant_state_guess(self):
        """Seed every collocation state with its initial condition.

        ``solve_local_nlp`` starts at the midpoint of the *clipped* bounds, so a
        state declared with the default ``ub=1e20`` starts at 50. For a stiff
        system that is catastrophic: a radical concentration whose true scale is
        1e-3 starts at 50, and a quadratic termination term ``2*kt*Y0**2``
        evaluates to 2.5e7 at the first iterate. A constant trajectory at the
        declared initial value costs nothing and is always in the right regime.
        """
        guess = {}
        shape = (self._nfe, self._ncp + 1)
        for block in (self._components, self._states):
            for name, elem in block.items():
                val = getattr(elem, 'value', None)
                if val is None:
                    continue
                guess[f't_{name}'] = np.full(shape, float(val))
        return guess

    # Warm-starting is best-effort: it must never become a new failure mode or
    # dominate the solve it is meant to help. A stiff model whose forward solve
    # is itself hard would otherwise pay for a near-hanging simulation before
    # every estimation.
    _WARM_START_MAX_ITER = 100

    def _simulate_for_warm_start(self):
        """Forward-solve the model at the initial parameter values.

        Returns ``(result, extracted)`` or ``None`` if the simulation fails or
        exceeds :attr:`_WARM_START_MAX_ITER`. On ``None`` the caller falls back
        to :meth:`constant_state_guess`.
        """
        try:
            m_sim, dae_sim = self.build_simulation_model()
            res = self.solve_local(
                m_sim, solver_options={'max_iter': self._WARM_START_MAX_ITER})
            if res.status not in ('optimal', 'feasible'):
                return None
            return res, self.extract_results(res, dae_sim, {})
        except Exception:
            return None

    def estimation_initial_guess(self):
        """Warm-start every variable block of the parameter-estimation NLP.

        ``solve_local_nlp`` otherwise starts at the midpoint of the *clipped*
        bounds. A concentration state declared with the default ``ub=1e20``
        clips to 100 and therefore starts at 50, while the true trajectory is
        O(1). Weighted by a small measurement variance (``1/sigma^2`` is 2.5e5
        at ``sigma^2 = 4e-6``) the solver diverges from there and POUNCE reports
        UNBOUNDED -- on an objective that is a sum of squares and cannot be
        unbounded below. Seeding the collocation blocks from a forward solve at
        the initial parameters removes the failure.

        Also seeds the bilinear spectral blocks: ``D = C(theta) S^T`` started
        from the bound midpoint converges to a poor local minimum, so ``S`` is
        seeded with the closed-form ``S = pinv(C) @ D``.

        Returns
        -------
        dict or None
            ``{variable_name: array}`` or ``None`` when the simulation fails.
        """
        sim = self._simulate_for_warm_start()
        if sim is None:
            return None
        res, extracted = sim

        # Collocation state blocks ('t_<name>', shape (nfe, ncp+1)). The
        # simulation and estimation models share names and shapes.
        guess = {name: np.asarray(val) for name, val in res.x.items()
                 if name != '_dummy_obj'}

        spectral = self._spectral_initial_guess_from(extracted)
        if spectral:
            guess.update(spectral)
        return guess

    def spectral_initial_guess(self):
        """Warm-start for a spectral fit's ``S_matrix`` / ``C_spectral`` blocks.

        Kept for backwards compatibility; prefer
        :meth:`estimation_initial_guess`, which also seeds the collocation
        state blocks.
        """
        sim = self._simulate_for_warm_start()
        if sim is None:
            return None
        return self._spectral_initial_guess_from(sim[1])

    def _spectral_initial_guess_from(self, extracted):
        """Seed ``S_matrix`` / ``C_spectral`` (and interaction blocks) from a
        simulated trajectory."""
        sd = self._spectral_data
        if sd is None or sd.D is None:
            return None
        abs_comps = self._absorbing_names()
        if not abs_comps:
            return None
        Z = extracted.get('Z')
        X = extracted.get('X')
        if Z is None or Z.empty:
            return None

        times = sd.times
        wavelengths = sd.wavelengths
        t_arr = np.array(Z.index, dtype=float)
        nearest = [int(np.argmin(np.abs(t_arr - float(tm)))) for tm in times]

        C_sim = np.zeros((len(times), len(abs_comps)))
        for ci, comp in enumerate(abs_comps):
            if comp not in Z.columns:
                return None
            C_sim[:, ci] = Z[comp].values[nearest]

        D_vals = np.asarray(sd.D.loc[times, wavelengths].values, dtype=float)
        D_filled = np.where(np.isfinite(D_vals), D_vals, 0.0)

        int_names = list(self._interactions)
        if not int_names:
            S0 = np.linalg.pinv(C_sim) @ D_filled        # (n_abs, n_lam)
            return {'S_matrix': S0, 'C_spectral': C_sim}

        try:
            int_cols = self._interaction_columns_numeric(Z, X)
        except Exception:
            return {'S_matrix': np.linalg.pinv(C_sim) @ D_filled,
                    'C_spectral': C_sim}

        Ci_sim = np.column_stack([np.atleast_1d(int_cols[nm])[nearest]
                                  for nm in int_names])

        # Seed both spectra blocks from one joint least-squares solve, so the
        # interaction spectra start at their conditional optimum rather than at
        # the midpoint of a +/-1e10 box.
        G = np.hstack([C_sim, Ci_sim])                   # (n_times, n_abs+n_int)
        S_all = np.linalg.pinv(G) @ D_filled
        return {'S_matrix': S_all[:len(abs_comps)],
                'C_spectral': C_sim,
                'S_interaction': S_all[len(abs_comps):],
                'C_interaction': Ci_sim}

    def _find_nearest_collocation_point(self, dae, t_meas):
        """Find the (i, k) index of the collocation point nearest to t_meas."""
        tp = dae._element_points()  # (nfe, ncp+1)
        best_dist = float('inf')
        best_i, best_k = 0, 0
        for i in range(tp.shape[0]):
            for k in range(tp.shape[1]):
                dist = abs(tp[i, k] - t_meas)
                if dist < best_dist:
                    best_dist = dist
                    best_i, best_k = i, k
        return best_i, best_k

    def _measurement_interpolation(self, dae, t_meas):
        """Lagrange-interpolation weights for a state at an arbitrary time.

        Returns ``(elem, coeffs)`` such that the state at ``t_meas`` is
        ``sum_k coeffs[k] * state[elem, k]`` over the ``ncp + 1`` nodes of the
        containing element.  This evaluates the collocation polynomial at the
        *exact* measurement time instead of snapping to the nearest node, which
        removes the time-misalignment bias that node-snapping introduces at
        coarse discretizations (see discopt#94).

        Parameters
        ----------
        dae : DAEBuilder
        t_meas : float

        Returns
        -------
        (int, numpy.ndarray)
            Element index and length-``(ncp + 1)`` interpolation coefficients.
        """
        from discopt.dae.polynomials import lagrange_basis

        cs = dae._cs
        tp = dae._element_points()  # (nfe, ncp+1)
        if cs.element_boundaries is not None:
            eb = np.asarray(cs.element_boundaries, dtype=np.float64)
        else:
            eb = np.linspace(cs.bounds[0], cs.bounds[1], cs.nfe + 1)

        t = float(t_meas)
        # Element containing t: largest i with eb[i] <= t, clamped to a valid
        # element so endpoint measurements land in the last element.
        elem = int(np.clip(np.searchsorted(eb, t, side='right') - 1, 0, cs.nfe - 1))

        # lagrange_basis is affine-invariant in the abscissa, so the element's
        # actual node times can be used directly as interpolation nodes.
        nodes = tp[elem, :]
        coeffs = np.array([lagrange_basis(nodes, t, k) for k in range(cs.ncp + 1)],
                          dtype=np.float64)
        return elem, coeffs

    def _build_concentration_objective(self, m, dae, variances, prefix=''):
        """
        Build NLS residual terms for concentration measurements.

        For each (t, component) measurement pair:
          (state[i,k] - C_meas)^2 / sigma_comp

        Parameters
        ----------
        prefix : str, optional
            Name prefix for the DAE state lookups. Empty for a single model;
            ``ReactionSet`` passes ``'<experiment>_'`` because every experiment's
            states live in one shared discopt model.
        """
        import discopt.modeling as dm

        c_data = self._get_concentration_data()
        if not c_data:
            return []

        obj_terms = []

        for comp_name, df in c_data.items():
            sigma = variances.get(comp_name, 1.0)
            if sigma is None or sigma <= 0:
                sigma = 1.0

            state_var = dae.get_state(prefix + comp_name)

            for t_meas, row in df.iterrows():
                c_meas = float(row.iloc[0])
                if np.isnan(c_meas):
                    continue
                # Interpolate the state to the exact measurement time rather
                # than snapping to the nearest collocation node (see discopt#94).
                elem, coeffs = self._measurement_interpolation(dae, float(t_meas))
                c_sim = coeffs[0] * state_var[elem, 0]
                for k in range(1, self._ncp + 1):
                    c_sim = c_sim + coeffs[k] * state_var[elem, k]
                resid = (c_sim - c_meas) ** 2 / sigma
                obj_terms.append(resid)

        return obj_terms

    def _build_spectral_objective(self, m, dae, dm_params, variances, prefix='',
                                  shared_S=None):
        """
        Build the NLS residual for spectral (Beer-Lambert) data:
        ``sum_{t,lam} (D[t,lam] - sum_c C[t,c]*S[c,lam])^2 / sigma_device``.

        ``prefix`` namespaces both the DAE state lookups and the spectral
        decision variables (``S_matrix``, ``C_spectral``, ...). It is empty for a
        single model; ``ReactionSet`` passes ``'<experiment>_'`` so that each
        experiment gets its own ``S`` and ``C`` blocks inside one shared discopt
        model.

        By default each experiment keeps a *separate* ``S``: a shared
        pure-component spectrum is only valid when the optical configuration is
        identical, and band shapes generally drift with temperature, so tying
        them is a modelling choice the caller must make explicitly rather than a
        default. Pass ``shared_S`` (a discopt variable of shape
        ``(n_abs, n_lam)``) to override -- ``ReactionSet.run_opt(shared_spectra=True)``
        builds one and hands the same variable to every member. When the members
        really are the same chemistry at different conditions, sharing removes
        ``(n_experiments - 1) * n_abs * n_lam`` degrees of freedom that would
        otherwise let each experiment explain away the very differences between
        conditions that identify the kinetics.

        The objective is built as a single vectorized matmul ``C @ S`` rather
        than a Python loop over the ``n_times * n_lam`` bilinear terms. The loop
        form produces a huge scalar expression whose JAX Hessian compilation is
        super-linear in problem size (tens of seconds to minutes at realistic
        sizes -- it dominates every spectral solve); the matmul form compiles in
        ~0.1s and is flat in size. ``S_matrix`` is stored as ``(n_abs, n_lam)``
        so ``C @ S`` needs no transpose (discopt Variables have none).
        """
        import discopt.modeling as dm

        D = self._spectral_data.D
        times = self._spectral_data.times
        wavelengths = self._spectral_data.wavelengths
        n_times = len(times)
        n_lam = len(wavelengths)

        # Absorbing components
        abs_comps = self._absorbing_names()

        n_abs = len(abs_comps)
        if n_abs == 0:
            return []

        int_names = list(self._interactions)
        n_int = len(int_names)

        # Pure-component spectra, stored (n_abs, n_lam) so D_pred = C @ S.
        #
        # Species spectra are physically non-negative, so the default lower
        # bound is 0. That is a statement about RAW absorbance: the standard
        # chemometric front-end breaks it. MSC applies a per-spectrum AFFINE map
        # (subtract an intercept, divide by a slope), and an affine map is not
        # Beer-Lambert-preserving, so the pure-component spectra of
        # baseline-corrected data genuinely have negative regions. Imposing
        # S >= 0 there is a false constraint that biases the kinetics, which is
        # what ``spectra_nonneg=False`` exists to switch off. Note that C_var
        # below keeps its lower bound of 0 either way -- concentrations really
        # are non-negative no matter what was done to the spectra.
        if shared_S is not None:
            S_var = shared_S           # one S tied across a ReactionSet
        else:
            S_var = m.continuous(prefix + 'S_matrix', shape=(n_abs, n_lam),
                                 lb=0.0 if self._spectra_nonneg else -1e10,
                                 ub=1e10)

        # C at measurement times: shape (n_times, n_abs)
        C_var = m.continuous(prefix + 'C_spectral', shape=(n_times, n_abs),
                             lb=0.0, ub=1e10)

        # Link C_var[ti, ci] to the state interpolated to the exact measurement
        # time rather than the nearest collocation node (see discopt#94).
        def interp(state_var, elem, coeffs):
            c = coeffs[0] * state_var[elem, 0]
            for k in range(1, self._ncp + 1):
                c = c + coeffs[k] * state_var[elem, k]
            return c

        subst_base = self._make_subst_base(dm_params)

        constraints = []
        int_rows = []
        for ti, t_meas in enumerate(times):
            elem, coeffs = self._measurement_interpolation(dae, float(t_meas))
            for ci, comp_name in enumerate(abs_comps):
                c_sim = interp(dae.get_state(prefix + comp_name), elem, coeffs)
                constraints.append(C_var[ti, ci] == c_sim)

            if n_int:
                # Bind every state name to its interpolated value at t_meas, then
                # evaluate the interaction expressions there. An interaction is an
                # ALGEBRAIC function of the states (e.g. c_A*c_B), not a
                # differential state -- giving it an ODE via the product rule
                # would add a spurious quadratic differential constraint.
                subst = dict(subst_base)
                for name in list(self._components) + list(self._states):
                    subst[name] = interp(dae.get_state(prefix + name), elem, coeffs)
                int_rows.append([resolve_expression(self._interactions[nm], subst)
                                 for nm in int_names])

        if constraints:
            m.subject_to(constraints, name=prefix + 'spectral_C_link')

        # Device variance
        sigma_device = variances.get('device', 1.0)
        if sigma_device is None or sigma_device <= 0:
            sigma_device = 1.0

        # Vectorized Beer-Lambert residual: D_pred = C @ S  (n_times x n_lam).
        D_vals = np.asarray(D.loc[times, wavelengths].values, dtype=float)
        finite = np.isfinite(D_vals)
        R = C_var @ S_var - np.where(finite, D_vals, 0.0)

        obj_extra = []
        if n_int:
            # Interaction spectra may be SIGNED: a band shift contributes a
            # derivative spectrum -delta*s'(nu), negative on one lobe.
            Si_var = m.continuous(prefix + 'S_interaction', shape=(n_int, n_lam),
                                  lb=-1e10, ub=1e10)
            Ci_var = m.continuous(prefix + 'C_interaction', shape=(n_times, n_int),
                                  lb=-1e10, ub=1e10)
            link = []
            for ti in range(n_times):
                for j in range(n_int):
                    link.append(Ci_var[ti, j] == int_rows[ti][j])
            m.subject_to(link, name=prefix + 'spectral_C_interaction_link')

            R = R + Ci_var @ Si_var

            if self._spectral_ridge > 0:
                # The augmented model is more flexible and can trade interaction
                # spectra against the rate constants. A ridge keeps it identified.
                obj_extra.append(self._spectral_ridge * dm.sum(Si_var * Si_var))

        if not finite.all():
            # Drop missing measurements from the objective.
            R = R * finite.astype(float)
        return [dm.sum(R * R) / sigma_device] + obj_extra

    # ── Results extraction ──

    def extract_results(self, result, dae, dm_params, prefix='', s_prefix=None):
        """
        Extract solution from a discopt SolveResult.

        Parameters
        ----------
        prefix : str, optional
            Name prefix used when the states were declared. Empty for a single
            model; ``'<experiment>_'`` for a ``ReactionSet`` member.
        s_prefix : str, optional
            Prefix for the ``S_matrix`` lookup alone. Defaults to ``prefix``;
            ``ReactionSet`` passes ``''`` when the species spectra are shared
            across experiments, since that block is declared once and unprefixed.
            Every member then reports the same ``S``, which is the point.

        Returns
        -------
        dict with keys:
          'Z': DataFrame of component trajectories (time x components)
          'X': DataFrame of state trajectories (time x states)
          'Y': DataFrame of algebraic trajectories (time x algebraics)
          'P': dict of parameter values
          'S': DataFrame of S matrix (wavelengths x abs_components), if spectral
        """
        import pandas as pd

        # Extract time axis
        tp = dae._element_points()  # (nfe, ncp+1)
        t_list = []
        idx_list = []
        for i in range(self._nfe):
            for k in range(self._ncp + 1):
                if i > 0 and k == 0:
                    continue
                t_list.append(tp[i, k])
                idx_list.append((i, k))

        # Extract component concentrations (Z)
        Z_data = {}
        for comp_name in self._components:
            var = dae.get_state(prefix + comp_name)
            val = result.value(var)
            Z_data[comp_name] = [float(val[i, k]) for i, k in idx_list]
        Z = pd.DataFrame(Z_data, index=t_list)

        # Extract non-concentration states (X)
        X_data = {}
        for state_name in self._states:
            var = dae.get_state(prefix + state_name)
            val = result.value(var)
            X_data[state_name] = [float(val[i, k]) for i, k in idx_list]
        X = pd.DataFrame(X_data, index=t_list) if X_data else pd.DataFrame(index=t_list)

        # Extract algebraics (Y) - at collocation points only
        Y_data = {}
        tp2 = dae._element_points()
        t_alg = []
        alg_idx = []
        for i in range(self._nfe):
            for j in range(self._ncp):
                t_alg.append(tp2[i, j + 1])
                alg_idx.append((i, j))

        for alg_name in self._algebraics:
            var = dae._vars[prefix + alg_name]
            val = result.value(var)
            Y_data[alg_name] = [float(val[i, j]) for i, j in alg_idx]
        Y = pd.DataFrame(Y_data, index=t_alg) if Y_data else pd.DataFrame(index=t_alg)

        # Extract parameters.
        #
        # dm_params maps a name to either a plain number (a fixed parameter
        # substituted as a constant) or a scalar discopt Variable; dispatch on
        # which it is. The previous guard tested
        # ``hasattr(dm_var, '__len__') is False``, but discopt Variables define
        # __len__ whatever their shape, so it never fired and every estimated
        # parameter came back as the unevaluated Variable object.
        P = {}
        for name, p in self._parameters.items():
            if p.fixed:
                P[name] = p.value
                continue

            dm_var = dm_params.get(name)
            if dm_var is None:
                P[name] = p.value
            elif isinstance(dm_var, (int, float, np.integer, np.floating)):
                P[name] = float(dm_var)
            else:
                try:
                    P[name] = float(np.asarray(result.value(dm_var)).reshape(()))
                except Exception:
                    P[name] = p.value

        # Extract S matrix if spectral
        S = None
        if self._spectral_data is not None and self._spectral_data.D is not None:
            abs_comps = self._absorbing_names()
            wavelengths = self._spectral_data.wavelengths
            try:
                s_var = result.x.get(
                    (prefix if s_prefix is None else s_prefix) + 'S_matrix')
                if s_var is not None:
                    # Stored (n_abs, n_lam); present as (wavelengths x components).
                    S = pd.DataFrame(np.asarray(s_var).T, index=wavelengths,
                                     columns=abs_comps)
                # Interaction spectra (Gamma) are extra columns of S.
                si_var = result.x.get(prefix + 'S_interaction')
                if si_var is not None and self._interactions:
                    Si = pd.DataFrame(np.asarray(si_var).T, index=wavelengths,
                                      columns=list(self._interactions))
                    S = Si if S is None else pd.concat([S, Si], axis=1)
            except Exception:
                pass

        return {'Z': Z, 'X': X, 'Y': Y, 'P': P, 'S': S}
