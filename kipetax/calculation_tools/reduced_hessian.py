"""
Analytic identifiability diagnostics for kipetax.

The legacy :mod:`kipetax.calculation_tools.covariance` estimates the parameter
covariance with a finite-difference Jacobian: each parameter is perturbed, the
simulation is **re-solved**, and the residuals are differenced.  That procedure
is slow and -- worse -- silently breaks down when a parameter is structurally
unidentifiable: ``J^T J`` becomes singular and the ``pinv`` fallback returns a
huge, meaningless confidence interval with no warning that the direction is
null.

This module replaces that with an *analytic* Fisher Information Matrix (FIM)
built from discopt's JAX-compiled constraint Jacobian, and exposes the rank /
null space of the FIM so structural unidentifiability is reported explicitly
rather than hidden inside an exploding interval.

Method (sensitivity-equation / Gauss-Newton FIM)
------------------------------------------------
The discretized estimation problem has primal variables ``x = [theta, z]``,
where ``theta`` are the unknown kinetic parameters and ``z`` are the collocation
state (and algebraic) variables, tied together by the DAE collocation equality
constraints ``c(x) = 0``.  By the implicit function theorem, along the
constraint manifold::

    J_z (dz/dtheta) + J_theta = 0   =>   dz/dtheta = -J_z^{-1} J_theta

where ``J = [J_theta | J_z]`` is the analytic constraint Jacobian at the
solution.  The sensitivity of a measured state output ``y`` to the parameters is
read straight off the rows of ``dz/dtheta`` at the measured collocation points.
The FIM is then the weighted Gram matrix::

    FIM = S^T W S,   S = dy/dtheta,   W = diag(1/sigma_component)

and ``Cov(theta) = FIM^{-1}`` (scaled by the reduced chi-square when the
variances are estimated rather than known).  Crucially, an eigendecomposition of
the FIM exposes structural unidentifiability directly: a near-zero eigenvalue is
a parameter combination the measurements cannot constrain -- the "null
direction" that, in the steady-state-vs-transient argument, is only identifiable
under transient excitation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IdentifiabilityReport:
    """Structural identifiability diagnostics from the Fisher Information Matrix.

    Attributes
    ----------
    param_names : list of str
        Estimated parameter names, in FIM row/column order.
    fim : numpy.ndarray
        The Fisher Information Matrix (n_params x n_params).
    eigenvalues : numpy.ndarray
        FIM eigenvalues, ascending. Near-zero entries are null directions.
    rank : int
        Numerical rank of the FIM (eigenvalues above ``rank_tol * lambda_max``).
    condition_number : float
        ``lambda_max / lambda_min``; ``inf`` when rank-deficient.
    identifiable : dict
        ``{name: bool}`` -- False when the parameter loads substantially on a
        null direction.
    null_space : list of dict
        One entry per null direction: ``{'eigenvalue': float,
        'combination': {name: weight}}`` giving the unidentifiable parameter
        combination (weights are the eigenvector components).
    warnings : list of str
        Human-readable warnings, one per unidentifiable direction.
    """

    param_names: list
    fim: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    identifiable: dict
    null_space: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def is_fully_identifiable(self) -> bool:
        return self.rank == len(self.param_names)

    def summary(self) -> str:
        n = len(self.param_names)
        lines = [
            "Identifiability report",
            f"  parameters    : {n}",
            f"  FIM rank      : {self.rank}/{n}"
            + ("" if self.is_fully_identifiable else "   <-- RANK DEFICIENT"),
            f"  FIM cond      : {self.condition_number:.4g}",
            f"  eigenvalues   : {np.array2string(self.eigenvalues, precision=4)}",
        ]
        for name in self.param_names:
            flag = "identifiable" if self.identifiable[name] else "NOT identifiable"
            lines.append(f"    {name:>12s} : {flag}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def _variable_offsets(model):
    """Map each discopt variable name to its start index in the flat vector."""
    offsets = {}
    o = 0
    for v in model._variables:
        offsets[v.name] = o
        o += v.size
    return offsets


def compute_fim(builder, model, dae, x_flat, variances, free_params=None,
                evaluator=None):
    """Analytic Fisher Information Matrix at ``x_flat``.

    Parameters
    ----------
    builder : DiscoptModelBuilder
        Builder that produced ``model`` (used for the data/measurement map and
        the collocation-point lookup).
    model : discopt.Model
        The estimation model after ``discretize()``.
    dae : DAEBuilder
        The DAE builder (collocation grid).
    x_flat : numpy.ndarray
        Flat primal vector (in ``model._variables`` order) at the solution.
    variances : dict
        Component measurement variances ``{name: sigma_sq}``.
    free_params : list of str, optional
        Names of the estimated (non-fixed) parameters. Defaults to every
        non-fixed parameter in ``builder``.
    evaluator : discopt NLPEvaluator, optional
        Evaluator for ``model`` to reuse (e.g. the one built for the solve)
        instead of compiling a fresh one.

    Returns
    -------
    (fim, param_names) : (numpy.ndarray, list of str)
    """
    from kipetax.model_tools.discopt_compat import get_nlp_evaluator
    NLPEvaluator = get_nlp_evaluator()

    if free_params is None:
        free_params = [name for name, p in builder._parameters.items() if not p.fixed]

    x_flat = np.asarray(x_flat, dtype=np.float64)
    n_vars = x_flat.size

    ev = evaluator if evaluator is not None else NLPEvaluator(model)
    J = ev.evaluate_jacobian(x_flat)              # (m_cons, n_vars), analytic
    lb, ub = ev.variable_bounds

    offsets = _variable_offsets(model)
    param_cols = [offsets[f"P_{name}"] for name in free_params]
    n_params = len(param_cols)

    # Partition columns: parameters / fixed (bounds pin them) / free states.
    fixed = np.abs(ub - lb) <= 1e-12
    is_param = np.zeros(n_vars, dtype=bool)
    is_param[param_cols] = True
    free_z = np.where(~fixed & ~is_param)[0]
    pos_in_z = {int(col): j for j, col in enumerate(free_z)}

    J_theta = J[:, param_cols]                    # (m_cons, n_params)
    J_z = J[:, free_z]                            # (m_cons, n_free_z)

    # dz/dtheta along the constraint manifold: solve J_z X = -J_theta.
    # lstsq handles the (square, well-posed) collocation case exactly and
    # degrades gracefully if the active set leaves the system non-square.
    # Variables that appear only in the objective (e.g. the spectral S_matrix)
    # contribute all-zero columns to J_z; lstsq min-norms their sensitivity to
    # zero, leaving the genuine state sensitivities intact.
    dz_dtheta, *_ = np.linalg.lstsq(J_z, -J_theta, rcond=None)   # (n_free_z, n_params)

    ncp1 = dae._cs.ncp + 1

    def sensitivity_row(comp_name, t_meas):
        """dy/dtheta for a state ``comp_name`` interpolated to ``t_meas``.

        The measured output is the state interpolated to the exact measurement
        time (matching the objective; see discopt#94), so its sensitivity is the
        same Lagrange-weighted combination of the element's node sensitivities.
        Returns an ``n_params`` vector, or ``None`` if the state is absent.
        """
        vname = f"{dae._cs.name}_{comp_name}"
        base = offsets.get(vname)
        if base is None:
            return None
        elem, coeffs = builder._measurement_interpolation(dae, float(t_meas))
        srow = np.zeros(n_params)
        for k in range(ncp1):
            full_idx = base + elem * ncp1 + k
            ck = coeffs[k]
            if full_idx in pos_in_z:
                srow = srow + ck * dz_dtheta[pos_in_z[full_idx], :]
            elif full_idx in param_cols:
                srow[param_cols.index(full_idx)] += ck
            # a node pinned by a bound contributes zero sensitivity
        return srow

    fim = np.zeros((n_params, n_params))

    # --- Directly observed concentrations: y = state ---
    for comp_name, df in builder._get_concentration_data().items():
        sigma = variances.get(comp_name, 1.0)
        if sigma is None or sigma <= 0:
            sigma = 1.0
        for t_meas, row in df.iterrows():
            c_meas = float(row.iloc[0])
            if np.isnan(c_meas):
                continue
            srow = sensitivity_row(comp_name, float(t_meas))
            if srow is None:
                continue
            fim = fim + np.outer(srow, srow) / sigma

    # --- Spectral observations: D = C(theta) S^T, free spectra S profiled out ---
    if builder._spectral_data is not None and builder._spectral_data.D is not None:
        fim = fim + _spectral_fim(builder, dae, x_flat, offsets, variances,
                                  sensitivity_row, n_params)

    return fim, free_params


def _spectral_fim(builder, dae, x_flat, offsets, variances, sensitivity_row,
                  n_params):
    """FIM contribution from spectral data ``D = C(theta) S^T``.

    The measured output is ``D[t, lam] = sum_c C[t,c] S[lam,c]``. The
    pure-component spectra ``S`` are a large block of free variables determined
    only by the least-squares objective, not by any equality constraint, so they
    must be *profiled out* rather than treated as constraint-determined states
    (the latter makes ``J_z`` rank-deficient and misreports the unconstrained-S
    artifact as unidentifiability of the kinetic parameters).

    For fixed ``C`` the optimal ``S`` is linear, so the separable / VARPRO
    Gauss-Newton information for ``theta`` is

        FIM = (1/sigma) * sum_lam V_lam^T (I - P_C) V_lam,
        V_lam[t, p] = sum_c (dC[t,c]/dtheta_p) * S[lam,c],

    where ``P_C = C (C^T C)^+ C^T`` projects onto ``range(C)`` over the
    measurement-time index. A ``theta`` direction whose output sensitivity lies
    entirely in ``range(C)`` is absorbed by re-fitting ``S`` and is genuinely
    unidentifiable from spectral data; everything orthogonal to it is informative
    and yields a finite covariance.
    """
    sd = builder._spectral_data
    times = sd.times
    wavelengths = sd.wavelengths
    n_times, n_lam = len(times), len(wavelengths)

    # Absorbing components, in the order used to build C_spectral / S_matrix.
    if builder._absorbing_components is not None:
        abs_comps = builder._absorbing_components
    else:
        abs_comps = [name for name in builder._components
                     if builder._components[name].absorbing
                     and name not in builder._non_absorbing]
    n_abs = len(abs_comps)
    if n_abs == 0 or n_times == 0 or n_lam == 0:
        return np.zeros((n_params, n_params))

    s_base = offsets.get('S_matrix')
    c_base = offsets.get('C_spectral')
    if s_base is None or c_base is None:
        return np.zeros((n_params, n_params))

    # Optimal spectra and concentrations from the flat solution (row-major,
    # matching extract_x_flat's ravel of the (n_lam, n_abs) / (n_times, n_abs)
    # variables).
    S_opt = x_flat[s_base:s_base + n_abs * n_lam].reshape(n_abs, n_lam)
    C_opt = x_flat[c_base:c_base + n_times * n_abs].reshape(n_times, n_abs)

    # Concentration sensitivities G[ti, ci, :] = dC[ci](t_i)/dtheta.
    G = np.zeros((n_times, n_abs, n_params))
    for ci, comp_name in enumerate(abs_comps):
        for ti, t_meas in enumerate(times):
            srow = sensitivity_row(comp_name, float(t_meas))
            if srow is not None:
                G[ti, ci, :] = srow

    sigma = variances.get('device', 1.0)
    if sigma is None or sigma <= 0:
        sigma = 1.0

    CtC_pinv = np.linalg.pinv(C_opt.T @ C_opt)    # (n_abs, n_abs)

    fim = np.zeros((n_params, n_params))
    for li in range(n_lam):
        # V[t, p] = sum_c G[t, c, p] * S_opt[c, li]
        V = np.tensordot(G, S_opt[:, li], axes=([1], [0]))      # (n_times, n_params)
        # Project range(C) out over the time index: Vp = (I - P_C) V.
        Vp = V - C_opt @ (CtC_pinv @ (C_opt.T @ V))            # (n_times, n_params)
        fim = fim + (Vp.T @ Vp) / sigma
    return fim


def assess_identifiability(fim, param_names, rank_tol=1e-8, loading_tol=0.5):
    """Eigen-decompose the FIM and flag structurally unidentifiable directions.

    Parameters
    ----------
    fim : numpy.ndarray
        Fisher Information Matrix (n x n, symmetric PSD).
    param_names : list of str
        Parameter names in FIM order.
    rank_tol : float
        An eigenvalue is treated as zero (a null direction) when
        ``lambda_i < rank_tol * lambda_max``.
    loading_tol : float
        A parameter is flagged unidentifiable when the absolute weight of any
        null eigenvector on it exceeds ``loading_tol`` (eigenvectors are unit
        norm, so this is a fraction of a fully-concentrated direction).

    Returns
    -------
    IdentifiabilityReport
    """
    fim = np.asarray(fim, dtype=np.float64)
    n = len(param_names)
    # Symmetric eigendecomposition (FIM is symmetric PSD by construction).
    evals, evecs = np.linalg.eigh(0.5 * (fim + fim.T))
    evals = np.clip(evals, 0.0, None)             # kill tiny negative round-off
    lam_max = float(evals[-1]) if n else 0.0

    thresh = rank_tol * lam_max if lam_max > 0 else 0.0
    null_mask = evals <= thresh
    rank = int(np.count_nonzero(~null_mask))

    lam_min_nonzero = evals[~null_mask].min() if rank else 0.0
    if rank == 0:
        condition_number = float("inf")
    elif null_mask.any():
        condition_number = float("inf")
    else:
        condition_number = float(lam_max / lam_min_nonzero) if lam_min_nonzero > 0 else float("inf")

    identifiable = {name: True for name in param_names}
    null_space = []
    warnings = []
    for idx in np.where(null_mask)[0]:
        vec = evecs[:, idx]
        combination = {name: float(vec[j]) for j, name in enumerate(param_names)}
        null_space.append({"eigenvalue": float(evals[idx]), "combination": combination})
        loaded = [name for j, name in enumerate(param_names) if abs(vec[j]) >= loading_tol]
        for name in loaded:
            identifiable[name] = False
        combo_str = " + ".join(f"{vec[j]:+.3f}*{name}" for j, name in enumerate(param_names))
        if len(loaded) > 1:
            warnings.append(
                f"Parameters {{{', '.join(loaded)}}} are not separately identifiable "
                f"from this data: only the combination ({combo_str}) is constrained. "
                f"They are identifiable only under transient excitation."
            )
        elif loaded:
            warnings.append(
                f"Parameter '{loaded[0]}' is structurally unidentifiable from this "
                f"data (null direction {combo_str}); identifiable only under "
                f"transient excitation."
            )
        else:
            warnings.append(
                f"Unidentifiable parameter combination ({combo_str}); "
                f"identifiable only under transient excitation."
            )

    return IdentifiabilityReport(
        param_names=list(param_names),
        fim=fim,
        eigenvalues=evals,
        rank=rank,
        condition_number=condition_number,
        identifiable=identifiable,
        null_space=null_space,
        warnings=warnings,
    )


def covariance_from_fim(fim, report, scale=1.0):
    """Parameter covariance from the FIM, null directions set to infinite variance.

    For identifiable directions ``Cov = scale * FIM^{-1}``. Each null direction
    contributes infinite variance, so the corresponding diagonal entries are
    ``inf`` rather than a meaningless large finite number.

    Parameters
    ----------
    fim : numpy.ndarray
    report : IdentifiabilityReport
    scale : float
        Variance scale (``1.0`` when measurement variances are known; otherwise
        the reduced chi-square ``SSE / (n_meas - n_params)``).

    Returns
    -------
    numpy.ndarray
        Covariance matrix. Unidentifiable parameters have ``inf`` on the diagonal.
    """
    fim = np.asarray(fim, dtype=np.float64)
    n = fim.shape[0]
    evals, evecs = np.linalg.eigh(0.5 * (fim + fim.T))
    evals = np.clip(evals, 0.0, None)
    lam_max = float(evals[-1]) if n else 0.0
    thresh = 1e-8 * lam_max if lam_max > 0 else 0.0

    cov = np.zeros((n, n))
    has_null = False
    for i in range(n):
        if evals[i] <= thresh:
            has_null = True
            continue
        v = evecs[:, i]
        cov += scale * np.outer(v, v) / evals[i]

    if has_null:
        # Parameters loading on a null direction get infinite variance.
        for name, ok in report.identifiable.items():
            if not ok:
                j = report.param_names.index(name)
                cov[j, j] = np.inf
    return cov


def analytic_covariance(builder, model, dae, result, variances,
                        rank_tol=1e-8, scale=1.0, free_params=None,
                        evaluator=None):
    """End-to-end analytic covariance + identifiability from a solved model.

    Parameters
    ----------
    builder : DiscoptModelBuilder
    model : discopt.Model
        The estimation model (already solved to produce ``result``).
    dae : DAEBuilder
    result : discopt.SolveResult
        Solved result; ``result.x`` provides the converged primal values.
    variances : dict
        Component measurement variances.
    rank_tol, scale, free_params
        See :func:`assess_identifiability` and :func:`compute_fim`.

    Returns
    -------
    dict with keys ``covariance``, ``fim``, ``report`` (IdentifiabilityReport),
    ``param_names``, and ``cov_source`` (``'reduced_hessian'`` or
    ``'reduced_hessian(rank-deficient)'``).
    """
    from kipetax.model_tools.discopt_compat import get_extract_x_flat
    extract_x_flat = get_extract_x_flat()

    x_flat = np.asarray(extract_x_flat(result, model))
    fim, param_names = compute_fim(builder, model, dae, x_flat, variances,
                                   free_params=free_params, evaluator=evaluator)
    report = assess_identifiability(fim, param_names, rank_tol=rank_tol)
    cov = covariance_from_fim(fim, report, scale=scale)
    source = "reduced_hessian" if report.is_fully_identifiable \
        else "reduced_hessian(rank-deficient)"
    return {
        "covariance": cov,
        "fim": fim,
        "report": report,
        "param_names": param_names,
        "cov_source": source,
    }
