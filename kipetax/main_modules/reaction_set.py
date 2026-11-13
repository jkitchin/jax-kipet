"""
ReactionSet for kipetax (Pyomo-free).

Manages multiple ReactionModel instances for combined parameter estimation
across multiple experiments with shared parameters.
"""
from kipetax.main_modules.reaction_model import (
    ReactionModel, _nonneg_is_active)
from kipetax.estimator_tools.results_object import ResultsObject


class ReactionSet:
    """
    Container for multiple ReactionModel instances.

    Supports combined parameter estimation where certain parameters
    are shared across experiments.

    Parameters
    ----------
    name : str, optional
        Name for the combined model.
    """

    def __init__(self, name='reaction_set'):
        self.name = name
        self._models = {}         # name -> ReactionModel
        self._shared_params = {}  # param_name -> initial value
        self.results = ResultsObject()

    def __getitem__(self, name):
        return self._models[name]

    def __iter__(self):
        return iter(self._models.values())

    def __len__(self):
        return len(self._models)

    def add_model(self, model):
        """
        Add a ReactionModel to the set.

        Parameters
        ----------
        model : ReactionModel
        """
        if not isinstance(model, ReactionModel):
            raise TypeError("Only ReactionModel instances can be added")
        self._models[model.name] = model

    def set_shared_parameter(self, param_name, value=None, bounds=(None, None)):
        """
        Declare a parameter as shared across all experiments.

        Parameters
        ----------
        param_name : str
        value : float, optional
        bounds : tuple, optional
        """
        self._shared_params[param_name] = {'value': value, 'bounds': bounds}

    def _experiment_variances(self, variances, exp_name):
        """Flat variances, overridden by any per-experiment sub-dict.

        ``variances`` may be flat (``{'device': 1e-4}``, applied to every
        experiment) or carry a nested override keyed by experiment name
        (``{'device': 1e-4, 'run_523K': {'device': 4e-4}}``). Different
        temperatures rarely share a noise level, so the override matters.
        """
        merged = {k: v for k, v in variances.items()
                  if not isinstance(v, dict)}
        override = variances.get(exp_name)
        if isinstance(override, dict):
            merged.update(override)
        return merged

    def _validate_shared_spectra(self):
        """Check that one ``S`` can legitimately describe every experiment.

        Sharing ``S`` means the same matrix multiplies every experiment's ``C``,
        so the wavelength grid, the absorbing-species list and its ORDER must
        agree exactly. Returns ``(n_abs, n_lam)``.
        """
        import numpy as np

        ref_name = ref_lam = ref_abs = None
        for name, rm in self._models.items():
            b = rm._make_builder()
            sd = b._spectral_data
            if sd is None or sd.D is None:
                raise ValueError(
                    f"shared_spectra=True requires spectral data in every "
                    f"experiment, but {name!r} has none")
            lam = np.asarray(sd.wavelengths, dtype=float)
            abs_names = list(b._absorbing_names())
            if ref_lam is None:
                ref_name, ref_lam, ref_abs = name, lam, abs_names
                continue
            if lam.shape != ref_lam.shape or not np.allclose(lam, ref_lam):
                raise ValueError(
                    f"shared_spectra=True requires an identical wavelength "
                    f"grid; {name!r} has {lam.shape[0]} wavelengths vs "
                    f"{ref_lam.shape[0]} in {ref_name!r}"
                    + ("" if lam.shape != ref_lam.shape else
                       " (same count, different values)"))
            if abs_names != ref_abs:
                raise ValueError(
                    f"shared_spectra=True requires the same absorbing species "
                    f"in the same order; {name!r} has {abs_names} vs "
                    f"{ref_abs} in {ref_name!r}")
        return len(ref_abs), len(ref_lam)

    def run_opt(self, variances=None, solver_options=None, covariance=False,
                shared_spectra=False, spectra_nonneg=True):
        """
        Run combined parameter estimation across all experiments.

        Builds a single discopt model that aggregates the NLS objectives from
        all individual experiments, sharing specified parameters. Each
        experiment contributes whatever data it carries: a spectral D-matrix, a
        concentration table, or both.

        Parameters
        ----------
        variances : dict, optional
            Variance estimates. Flat entries apply to every experiment; a
            nested dict keyed by experiment name overrides them for that
            experiment. See :meth:`_experiment_variances`.
        solver_options : dict, optional
        covariance : bool, optional
            Not implemented for combined estimation; a warning is emitted if
            requested.
        shared_spectra : bool, optional
            Tie the pure-component spectra ``S`` across every experiment.

            Off by default, because it is only valid when the optical
            configuration is identical and the D-matrices are on a common
            absolute scale, and because band shapes drift with temperature --
            so it is a modelling choice, not a safe assumption.

            Turn it on when the members are *the same chemistry at different
            conditions* (an amplitude series, a feed series, a concentration
            series measured on one instrument). There the spectra are shared by
            construction, and giving each experiment its own ``S`` costs real
            identifiability: ``(n_experiments - 1) * n_abs * n_lam`` free
            parameters is enough for each experiment to explain away the very
            differences between conditions that identify the kinetics. A
            condition series where a rate constant varies by a fitted power law
            is the canonical case -- with free ``S`` per experiment the exponent
            can be badly biased while the objective barely moves.

            Requires every experiment to carry spectral data on an identical
            wavelength grid with an identical list of absorbing species; a
            :class:`ValueError` names the first mismatch otherwise.
        spectra_nonneg : bool or float, optional
            As in :meth:`ReactionModel.run_opt`. Relax it when the D-matrices
            have been baseline- or scatter-corrected, which makes the true
            pure-component spectra partly negative.

        Returns
        -------
        ResultsObject
            ``P`` holds the shared parameters plus each experiment's local ones
            (local names are prefixed ``<experiment>.``). ``experiments`` maps
            each experiment name to its own ``ResultsObject`` (``Z``, ``X``,
            ``Y``, ``S``, ``P``), which is also stored on the member model's
            ``results_dict['p_estimator']``.
        """
        import warnings

        import discopt.modeling as dm
        from discopt.dae.collocation import DAEBuilder
        from kipetax.model_tools.discopt_model_builder import (
            _balanced_sum, solve_local_nlp)
        from kipetax.model_tools.proxy_vars import resolve_expression

        if not self._models:
            raise ValueError("No models added to ReactionSet")

        if variances is None:
            variances = {}

        if covariance:
            warnings.warn(
                "ReactionSet.run_opt does not compute covariance yet; "
                "confidence intervals will be absent from the result.",
                stacklevel=2)

        # Identify shared parameter names (all params that appear in all models)
        if self._shared_params:
            shared_names = set(self._shared_params.keys())
        else:
            # Auto-detect: params present in all models
            sets = [set(m.parameters.names) for m in self._models.values()]
            shared_names = set.intersection(*sets) if sets else set()

        # Build combined discopt model
        m_combined = dm.Model(self.name)

        # One S block for the whole set, if asked for. Built up front so every
        # experiment's spectral objective can be handed the same variable.
        shared_S_var = None
        if shared_spectra:
            n_abs, n_lam = self._validate_shared_spectra()
            shared_S_var = m_combined.continuous(
                'S_matrix', shape=(n_abs, n_lam),
                lb=0.0 if _nonneg_is_active(spectra_nonneg) else -1e10,
                ub=1e10)

        # Initial-guess seeds for the POUNCE solve, keyed by variable name.
        # Scalars (parameter starting values) and whole variable blocks
        # (collocation states, spectral S/C) go through different channels.
        param_seed = {}
        block_seed = {}

        # Shared parameter variables
        shared_dm_params = {}
        for pname in shared_names:
            info = self._shared_params.get(pname, {})
            val = info.get('value')
            if val is None:
                # Fall back to the value declared on any model that has it, so
                # a shared parameter seeded only at the model level still gets a
                # sensible start. (`or 1.0` would silently discard a 0.0 seed.)
                for rm in self._models.values():
                    if pname in rm.parameters.names:
                        v = rm.parameters[pname].value
                        if v is not None:
                            val = v
                            break
            if val is None:
                val = 1.0
            bounds = info.get('bounds', (None, None))
            lb = bounds[0] if bounds[0] is not None else 0.0
            ub = bounds[1] if bounds[1] is not None else 1e10
            shared_dm_params[pname] = m_combined.continuous(f'P_{pname}', lb=lb, ub=ub)
            param_seed[f'P_{pname}'] = float(val)

        all_obj_terms = []
        contexts = []   # (exp_name, builder, dae, local_dm_params, prefix)

        for exp_name, rm in self._models.items():
            if rm._times is None:
                rm._infer_times_from_data()
            if rm._times is None:
                raise ValueError(f"Model {exp_name}: cannot determine time range")

            b = rm._make_builder()

            # Honour a graded mesh if the member model declared one.
            cs = b._continuous_set()
            dae = DAEBuilder(m_combined, cs)

            # Add states with experiment-prefixed names to avoid conflicts
            prefix = f'{exp_name}_'
            for comp in b._components.values():
                lb = comp.bounds[0] if comp.bounds[0] is not None else 0.0
                ub = comp.bounds[1] if comp.bounds[1] is not None else 1e20
                iv = comp.value if comp.value is not None else 0.0
                dae.add_state(prefix + comp.name, bounds=(lb, ub), initial=iv)

            for state in b._states.values():
                lb = state.bounds[0] if state.bounds[0] is not None else -1e20
                ub = state.bounds[1] if state.bounds[1] is not None else 1e20
                iv = state.value if state.value is not None else 0.0
                dae.add_state(prefix + state.name, bounds=(lb, ub), initial=iv)

            for alg in b._algebraics.values():
                lb = alg.bounds[0] if alg.bounds[0] is not None else -1e20
                ub = alg.bounds[1] if alg.bounds[1] is not None else 1e20
                dae.add_algebraic(prefix + alg.name, bounds=(lb, ub))

            # Local parameters (non-shared)
            local_dm_params = {}
            for pname, p in b._parameters.items():
                if pname in shared_names:
                    local_dm_params[pname] = shared_dm_params[pname]
                elif p.fixed:
                    # A fixed parameter is a constant, not a decision variable.
                    local_dm_params[pname] = p.value if p.value is not None else 0.0
                else:
                    lb = p.bounds[0] if p.bounds[0] is not None else 0.0
                    ub = p.bounds[1] if p.bounds[1] is not None else 1e10
                    local_dm_params[pname] = m_combined.continuous(f'{exp_name}_P_{pname}', lb=lb, ub=ub)
                    param_seed[f'{exp_name}_P_{pname}'] = (
                        float(p.value) if p.value is not None else 1.0)

            subst_base = {**local_dm_params}
            for cname, c in b._constants.items():
                subst_base[cname] = c.value if c.value is not None else 0.0

            # Remap ODE keys to prefixed state names
            odes_remapped = {}
            for var_name, expr in b._odes_dict.items():
                odes_remapped[prefix + var_name] = expr

            algs_remapped = {}
            for var_name, expr in b._algs_dict.items():
                algs_remapped[prefix + var_name] = expr

            def make_ode_rhs(odes, subst_b, pfx):
                def ode_rhs(t, states, algebraics, controls):
                    # Strip prefix from state names for expr lookup
                    states_stripped = {k[len(pfx):]: v for k, v in states.items()}
                    algs_stripped = {k[len(pfx):]: v for k, v in algebraics.items()}
                    subst = {**subst_b, **states_stripped, **algs_stripped}
                    result = {}
                    for vn, expr in odes.items():
                        result[vn] = resolve_expression(expr, subst)
                    return result
                return ode_rhs

            dae.set_ode(make_ode_rhs(odes_remapped, subst_base, prefix))

            if b._algs_dict:
                def make_alg_rhs(algs, subst_b, pfx):
                    def alg_rhs(t, states, algebraics, controls):
                        states_stripped = {k[len(pfx):]: v for k, v in states.items()}
                        algs_stripped = {k[len(pfx):]: v for k, v in algebraics.items()}
                        subst = {**subst_b, **states_stripped, **algs_stripped}
                        result = {}
                        for vn, expr in algs.items():
                            result[vn] = resolve_expression(expr, subst)
                        return result
                    return alg_rhs
                dae.set_algebraic(make_alg_rhs(algs_remapped, subst_base, prefix))

            dae.discretize()

            # Build objective terms for this experiment. Both blocks come from
            # the same builder methods the single-model path uses, with the
            # experiment prefix applied, so spectral and concentration data are
            # handled identically here and there.
            exp_var = self._experiment_variances(variances, exp_name)

            if b._spectral_data is not None and b._spectral_data.D is not None:
                b._spectra_nonneg = _nonneg_is_active(spectra_nonneg)
                all_obj_terms += b._build_spectral_objective(
                    m_combined, dae, local_dm_params, exp_var, prefix=prefix,
                    shared_S=shared_S_var)

            all_obj_terms += b._build_concentration_objective(
                m_combined, dae, exp_var, prefix=prefix)

            # Whole-block warm starts go through `initial_guess`, not
            # `parameter_values` -- the latter seeds one scalar per name.
            block_seed.update(
                self._warm_start_seed(rm, b, dae, shared_names, prefix,
                                      shared_spectra=shared_spectra))

            contexts.append((exp_name, b, dae, local_dm_params, prefix))

        # Set combined objective
        if all_obj_terms:
            m_combined.minimize(_balanced_sum(all_obj_terms))
        else:
            dummy = m_combined.continuous('_dummy', lb=0, ub=0)
            m_combined.minimize(dummy)

        # Solve -- always through POUNCE (never discopt's branch-and-bound).
        solve_kwargs = solver_options or {}
        result = solve_local_nlp(m_combined, parameter_values=param_seed,
                                 initial_guess=block_seed,
                                 solver_options=solve_kwargs)

        # Collect results
        ro = ResultsObject()
        ro.status = result.status
        ro.objective = result.objective

        ro.experiments = {}

        if result.status in ('optimal', 'feasible'):
            P = {}
            for pname, dm_var in shared_dm_params.items():
                try:
                    P[pname] = float(result.value(dm_var))
                except Exception:
                    P[pname] = None

            # Per-experiment trajectories, spectra and local parameters. Each
            # member model also gets its own ResultsObject so that
            # rm.results_dict['p_estimator'].S works exactly as after a
            # single-model run_opt.
            for exp_name, b, dae, local_dm_params, prefix in contexts:
                try:
                    extracted = b.extract_results(
                        result, dae, local_dm_params, prefix=prefix,
                        s_prefix='' if shared_spectra else None)
                except Exception as e:
                    print(f"Warning: could not extract results for "
                          f"{exp_name!r} ({e}).")
                    continue

                ro_i = ResultsObject()
                ro_i.load_from_builder_result(extracted, solve_result=result)
                ro.experiments[exp_name] = ro_i
                self._models[exp_name].results_dict['p_estimator'] = ro_i
                self._models[exp_name]._optimized = True

                # Local (non-shared) estimates are namespaced so they cannot
                # collide across experiments.
                for pname, pval in (extracted.get('P') or {}).items():
                    if pname not in shared_dm_params:
                        P[f'{exp_name}.{pname}'] = pval

            ro.P = P

        self.results = ro
        return ro

    def _warm_start_seed(self, model, builder, dae, shared_names, prefix,
                         shared_spectra=False):
        """Prefixed warm start for every variable block of one experiment.

        Combined estimation inherits the single-model failure mode that
        :meth:`DiscoptModelBuilder.estimation_initial_guess` exists to fix:
        without a seed, POUNCE starts each block at the midpoint of its clipped
        bounds and a variance-weighted objective diverges from there. It is
        worse here, because a bilinear ``S`` block per experiment is started
        from the middle of a +/-1e10 box.

        The member is simulated at the *shared* parameters' starting values
        rather than its own, which may be stale. Those values are restored
        afterwards: ``_make_builder`` hands out the very same
        ``ModelParameter`` objects the model holds, so mutating them in place
        would otherwise be a visible side effect of estimation.

        Returns
        -------
        dict
            ``{combined_variable_name: array}``, empty if the warm start fails.
        """
        saved = {}
        try:
            for pname in shared_names:
                info = self._shared_params.get(pname)
                if not info or info.get('value') is None:
                    continue
                if pname in builder._parameters:
                    p = builder._parameters[pname]
                    saved[pname] = p.value
                    p.value = info['value']

            guess = builder.estimation_initial_guess()
            if not guess:
                return {}

            # The simulation model names a state block 't_<name>'; the combined
            # model names the same block 't_<prefix><name>'. Rename by exact
            # substitution on the real variable name rather than by string
            # surgery on a guessed pattern -- state names can be suffixes of
            # one another ('A' and 'CA'), so endswith matching is not safe.
            rename = {}
            for name in list(builder._components) + list(builder._states):
                combined = dae.get_state(prefix + name).name
                rename[combined.replace(prefix + name, name, 1)] = combined
            for name in builder._algebraics:
                var = dae._vars.get(prefix + name)
                if var is not None:
                    combined = var.name
                    rename[combined.replace(prefix + name, name, 1)] = combined

            seed = {}
            for key, arr in guess.items():
                if key == '_dummy_obj':
                    continue
                if key == 'S_matrix' and shared_spectra:
                    # One unprefixed S block for the whole set. Every member
                    # proposes a seed for it and the last one wins, which is
                    # fine -- they are estimates of the same matrix. Leaving it
                    # unseeded is not: POUNCE would start the shared block at
                    # the middle of its +/-1e10 box, the failure mode this whole
                    # method exists to avoid.
                    seed['S_matrix'] = arr
                    continue
                # State/algebraic blocks are renamed; the spectral blocks
                # ('S_matrix', 'C_spectral', ...) just take the prefix.
                seed[rename.get(key, prefix + key)] = arr
            return seed
        except Exception as e:
            print(f"Warning: warm start failed for {model.name!r} ({e}); "
                  f"using solver defaults.")
            return {}
        finally:
            for pname, val in saved.items():
                builder._parameters[pname].value = val

    def __repr__(self):
        return f"ReactionSet({self.name!r}, models={list(self._models.keys())})"
