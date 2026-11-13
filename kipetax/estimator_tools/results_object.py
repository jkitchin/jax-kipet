"""
ResultsObject for kipetax.

Stores results from simulation, parameter estimation, and variance estimation.
"""
import numpy as np
import pandas as pd


class ResultsObject:
    """
    Container for kipetax estimation/simulation results.

    Attributes
    ----------
    P : dict
        Estimated parameter values {name: value}.
    Z : DataFrame
        Component concentration trajectories (time x components).
    X : DataFrame
        Non-concentration state trajectories (time x states).
    Y : DataFrame
        Algebraic variable trajectories (time x algebraics).
    S : DataFrame or None
        Molar absorptivity matrix (wavelengths x abs_components), if spectral.
    sigma_sq : dict
        Variance estimates {component: sigma_sq}.
    confidence_intervals : dict
        95% confidence intervals {param: (lower, upper)}. Unidentifiable
        parameters have ``(-inf, inf)`` rather than a spurious finite interval.
    covariance : numpy.ndarray or None
        Parameter covariance matrix (rows/cols ordered as ``P``). Diagonal
        entries are ``inf`` for structurally unidentifiable parameters.
    fim : numpy.ndarray or None
        Fisher Information Matrix at the solution.
    identifiability : IdentifiabilityReport or None
        Structural identifiability diagnostics (FIM rank, null space,
        per-parameter flags, warnings). See
        :mod:`kipetax.calculation_tools.reduced_hessian`.
    cov_source : str or None
        How the covariance was produced: ``'reduced_hessian'``,
        ``'reduced_hessian(rank-deficient)'``, or ``'finite_difference'``.
    status : str
        Solver status string.
    objective : float or None
        Objective value at solution.
    experiments : dict
        For a :class:`~kipetax.main_modules.reaction_set.ReactionSet` fit,
        ``{experiment_name: ResultsObject}`` holding each experiment's own
        trajectories and spectra. Empty for a single-model fit.
    """

    def __init__(self):
        self.P = {}
        self.Z = pd.DataFrame()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.S = None
        self.sigma_sq = {}
        self.confidence_intervals = {}
        self.covariance = None
        self.fim = None
        self.identifiability = None
        self.cov_source = None
        self.status = 'unknown'
        self.objective = None
        self.experiments = {}
        self._solver_result = None

    def load_from_builder_result(self, builder_result, solve_result=None):
        """
        Populate from the output of DiscoptModelBuilder.extract_results().

        Parameters
        ----------
        builder_result : dict
            Dict with keys 'Z', 'X', 'Y', 'P', 'S'.
        solve_result : discopt.SolveResult, optional
            The raw solver result for status/objective.
        """
        self.Z = builder_result.get('Z', pd.DataFrame())
        self.X = builder_result.get('X', pd.DataFrame())
        self.Y = builder_result.get('Y', pd.DataFrame())
        self.P = builder_result.get('P', {})
        self.S = builder_result.get('S', None)

        if solve_result is not None:
            self._solver_result = solve_result
            self.status = solve_result.status
            self.objective = solve_result.objective

    def load_identifiability(self, cov_result, alpha=0.05):
        """Populate covariance / identifiability fields and confidence intervals.

        Parameters
        ----------
        cov_result : dict
            Output of
            :func:`kipetax.calculation_tools.reduced_hessian.analytic_covariance`,
            with keys ``covariance``, ``fim``, ``report``, ``param_names``,
            ``cov_source``.
        alpha : float
            Significance level for the (two-sided normal) confidence intervals.
            ``0.05`` -> 95% intervals.
        """
        from scipy.stats import norm

        self.covariance = cov_result.get('covariance')
        self.fim = cov_result.get('fim')
        self.identifiability = cov_result.get('report')
        self.cov_source = cov_result.get('cov_source')

        param_names = cov_result.get('param_names', list(self.P.keys()))
        z = float(norm.ppf(1.0 - alpha / 2.0))
        if self.covariance is not None:
            for i, name in enumerate(param_names):
                var = float(self.covariance[i, i])
                p_val = self.P.get(name)
                if p_val is None:
                    continue
                if not np.isfinite(var):
                    # Structurally unidentifiable: unbounded interval, not a
                    # spurious huge finite one.
                    self.confidence_intervals[name] = (float('-inf'), float('inf'))
                else:
                    std = np.sqrt(max(var, 0.0))
                    self.confidence_intervals[name] = (p_val - z * std,
                                                       p_val + z * std)

    def compute_var_norm(self, var_name, order=np.inf):
        """
        Compute the norm of a trajectory variable (used in iterative variance estimation).

        Parameters
        ----------
        var_name : str
            One of 'Z', 'X', 'Y'.
        order : int or float
            Norm order.

        Returns
        -------
        float
        """
        df = getattr(self, var_name, pd.DataFrame())
        if df.empty:
            return 0.0
        return float(np.linalg.norm(df.values.ravel(), ord=order))

    def __repr__(self):
        params = ', '.join(f'{k}={v:.6g}' for k, v in self.P.items()) if self.P else 'none'
        return (f"ResultsObject(status={self.status!r}, "
                f"objective={self.objective}, "
                f"params=[{params}])")

    def __str__(self):
        s = f"ResultsObject\n"
        s += f"  Status    : {self.status}\n"
        s += f"  Objective : {self.objective}\n"
        if self.P:
            s += "  Parameters:\n"
            ident = self.identifiability
            for k, v in self.P.items():
                flag = ""
                if ident is not None and not ident.identifiable.get(k, True):
                    flag = "  [NOT identifiable]"
                ci = self.confidence_intervals.get(k)
                if ci:
                    s += (f"    {k:20s} = {v:.6g}  "
                          f"95% CI: ({ci[0]:.6g}, {ci[1]:.6g}){flag}\n")
                else:
                    s += f"    {k:20s} = {v:.6g}{flag}\n"
        if self.identifiability is not None:
            ident = self.identifiability
            n = len(ident.param_names)
            s += "  Identifiability:\n"
            s += (f"    FIM rank   : {ident.rank}/{n}"
                  + ("" if ident.is_fully_identifiable else "  (RANK DEFICIENT)")
                  + "\n")
            s += f"    FIM cond   : {ident.condition_number:.4g}\n"
            if self.cov_source:
                s += f"    cov source : {self.cov_source}\n"
            for w in ident.warnings:
                s += f"    ! {w}\n"
        if self.sigma_sq:
            s += "  Variances:\n"
            for k, v in self.sigma_sq.items():
                s += f"    {k:20s} = {v:.6g}\n"
        return s
