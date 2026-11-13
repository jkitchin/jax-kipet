"""
ParameterEstimator for kipetax.

Builds and solves the NLS parameter estimation problem using discopt.
"""
import numpy as np
import pandas as pd

from kipetax.estimator_tools.results_object import ResultsObject


class ParameterEstimator:
    """
    NLS parameter estimation using discopt collocation.

    Parameters
    ----------
    builder : DiscoptModelBuilder
        Populated builder object.
    """

    def __init__(self, builder):
        self._builder = builder

    def run_opt(self, variances=None, solver_options=None, covariance=False,
                cov_method='analytic'):
        """
        Run parameter estimation.

        The problem is always solved with a single local NLP solve (POUNCE
        interior point) -- the right tool for a nonconvex least-squares
        estimation and tractable at collocation sizes.

        Parameters
        ----------
        variances : dict, optional
            Component variances {name: sigma_sq}. Defaults to 1.0 for all.
        solver_options : dict, optional
            Extra keyword arguments forwarded to the POUNCE solver.
        covariance : bool, optional
            If True, compute confidence intervals and identifiability
            diagnostics.
        cov_method : {'analytic', 'pounce', 'auto', 'fd'}, optional
            Preferred covariance backend; each falls back to the next on
            failure so a covariance is still produced.

            * ``'analytic'`` (default) -- the Gauss-Newton reduced-Hessian /
              Fisher-Information-Matrix path
              (:mod:`kipetax.calculation_tools.reduced_hessian`), then FD. This
              is the statistically correct object for a parameter covariance:
              it is zero along a structurally unidentifiable direction, so the
              FIM eigendecomposition exposes that direction (infinite CI)
              instead of a spurious finite interval.
            * ``'pounce'`` / ``'auto'`` -- POUNCE's *full* reduced Hessian at
              the KKT point (the sIPOPT analog; reuses the solve's
              factorization), then analytic, then FD. NOTE: the full Lagrangian
              Hessian includes a residual-curvature (second-order) term the FIM
              drops, which agrees with the FIM for well-identified fits but adds
              spurious finite curvature to unidentifiable directions -- it can
              MASK structural unidentifiability. Prefer ``'analytic'`` for
              identifiability diagnostics. (A Gauss-Newton Hessian from POUNCE
              would remove this caveat; see discopt#98.)
            * ``'fd'`` -- the finite-difference Jacobian only.

        Returns
        -------
        ResultsObject
        """
        if variances is None:
            variances = {}

        m, dae, dm_params = self._builder.build_parameter_estimation_model(variances)

        solve_kwargs = {}
        if solver_options:
            solve_kwargs.update(solver_options)

        # Compile the evaluator once and reuse it for both the solve and the
        # analytic covariance (same model).
        from kipetax.model_tools.discopt_compat import get_nlp_evaluator
        NLPEvaluator = get_nlp_evaluator()
        evaluator = NLPEvaluator(m)

        # Seed every block from a forward solve at the initial parameters. The
        # bound-midpoint start puts collocation states at 50 (default ub=1e20
        # clips to 100), which diverges under a small measurement variance;
        # spectral fits additionally need S = pinv(C) @ D to avoid a poor local
        # minimum.
        initial_guess = self._builder.estimation_initial_guess()
        result = self._builder.solve_local(m, solver_options=solve_kwargs,
                                           initial_guess=initial_guess,
                                           evaluator=evaluator)

        ro = ResultsObject()
        if result.status in ('optimal', 'feasible'):
            extracted = self._builder.extract_results(result, dae, dm_params)
            ro.load_from_builder_result(extracted, result)

            if covariance:
                self._compute_confidence_intervals(
                    ro, variances, m, dae, result, method=cov_method,
                    evaluator=evaluator)
        else:
            ro.status = result.status
            ro.objective = result.objective
            print(f"Solver status: {result.status}")

        return ro

    def _reduced_chi_square_scale(self, ro, variances):
        """Variance scale for the covariance.

        ``1.0`` when measurement variances are supplied (known noise level);
        otherwise the reduced chi-square ``SSE / (n_meas - n_params)`` estimated
        from the objective.
        """
        if variances:
            return 1.0
        n_params = max(len(ro.P), 1)
        n_meas = 0
        for df in self._builder._get_concentration_data().values():
            n_meas += int(df.iloc[:, 0].notna().sum())
        sd = self._builder._spectral_data
        if sd is not None and sd.D is not None:
            n_meas += int(np.isfinite(sd.D.values).sum())
        dof = max(n_meas - n_params, 1)
        sse = float(ro.objective) if ro.objective is not None else 0.0
        return sse / dof

    def _compute_confidence_intervals(self, ro, variances, m, dae, result,
                                      method='analytic', evaluator=None):
        """Compute confidence intervals and identifiability diagnostics.

        ``method`` selects the preferred covariance backend; each falls back to
        the next on failure so a covariance is still produced:

        * ``'pounce'`` / ``'auto'`` -- POUNCE's full reduced Hessian at the KKT
          point (sIPOPT analog), then analytic FIM, then finite differences.
        * ``'analytic'`` (default) -- discopt-AD Gauss-Newton FIM, then FD.
        * ``'fd'`` -- finite-difference Jacobian only.
        """
        scale = self._reduced_chi_square_scale(ro, variances)
        if method in ('pounce', 'auto'):
            order = ['pounce', 'analytic', 'fd']
        elif method == 'fd':
            order = ['fd']
        else:
            order = ['analytic', 'fd']

        for backend in order:
            try:
                if backend == 'fd':
                    self._compute_confidence_intervals_fd(ro, variances)
                elif backend == 'pounce':
                    from kipetax.calculation_tools.pounce_covariance import pounce_covariance
                    ro.load_identifiability(pounce_covariance(
                        self._builder, m, dae, result, variances, scale=scale))
                else:  # 'analytic'
                    from kipetax.calculation_tools.reduced_hessian import analytic_covariance
                    ro.load_identifiability(analytic_covariance(
                        self._builder, m, dae, result, variances, scale=scale,
                        evaluator=evaluator))
                self._warn_if_degenerate(ro)
                return
            except Exception as e:
                nxt = order[order.index(backend) + 1] if backend != order[-1] else None
                msg = f"; falling back to {nxt}." if nxt else "."
                print(f"Warning: {backend} covariance failed ({e}){msg}")

    def _warn_if_degenerate(self, ro):
        """Warn loudly when a covariance could not be produced.

        A spectral fit that returns no covariance, or an all-infinite one, means
        the parameters could not be quantified -- surface that instead of
        leaving the caller to discover ``None`` / ``inf`` silently.
        """
        spectral = (self._builder._spectral_data is not None
                    and self._builder._spectral_data.D is not None)
        if ro.covariance is None:
            print(f"Warning: no parameter covariance could be computed"
                  f"{' for this spectral fit' if spectral else ''} "
                  f"(cov_source={ro.cov_source}).")
            return
        cov = np.asarray(ro.covariance, dtype=float)
        if cov.size and not np.any(np.isfinite(np.diag(cov))):
            print("Warning: parameter covariance is non-finite on every "
                  "parameter; the data does not determine the parameters "
                  "(check identifiability report).")

    def _compute_confidence_intervals_fd(self, ro, variances):
        """Legacy finite-difference Jacobian confidence intervals."""
        from kipetax.calculation_tools.covariance import compute_covariance
        # Use the same variance-scale convention as the analytic path so the two
        # methods agree: trust supplied variances (scale=1), otherwise estimate
        # the noise level from the residuals (scale=None -> reduced chi-square).
        scale = self._reduced_chi_square_scale(ro, variances)
        cov = compute_covariance(self._builder, ro.P, variances,
                                 scale=(scale if variances else None))
        if cov is not None:
            ro.covariance = cov
            ro.cov_source = 'finite_difference'
            param_names = list(ro.P.keys())
            for i, name in enumerate(param_names):
                std = float(np.sqrt(max(cov[i, i], 0)))
                p_val = ro.P[name]
                ro.confidence_intervals[name] = (p_val - 1.96 * std,
                                                  p_val + 1.96 * std)
