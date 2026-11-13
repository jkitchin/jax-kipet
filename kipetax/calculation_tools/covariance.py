"""
Covariance / confidence interval estimation for kipetax.

Uses finite-difference Jacobian to estimate Cov(P) after the NLS solve.
"""
import numpy as np


def compute_covariance(builder, P_opt, variances, eps_rel=1e-4, eps_abs=1e-8,
                       scale=None):
    """
    Estimate the covariance matrix of the estimated parameters via
    finite-difference Jacobian of the residual vector.

    For each parameter p_i, perturb by eps and re-solve, compute dR/dp_i.
    Then ``Cov(P) = scale * pinv(J^T @ J)``.

    Parameters
    ----------
    builder : DiscoptModelBuilder
        Builder with the problem specification.
    P_opt : dict
        Optimal parameter values {name: value}.
    variances : dict
        Component variances {name: sigma_sq}.
    eps_rel : float
        Relative perturbation size.
    eps_abs : float
        Absolute perturbation minimum.
    scale : float, optional
        Variance scale applied to ``pinv(J^T J)``. When ``None`` (default), the
        residual-estimated reduced chi-square ``SSE / (n_meas - n_params)`` is
        used. Pass ``1.0`` to trust the supplied ``variances`` as the absolute
        measurement variance -- this keeps the finite-difference covariance on
        the same scale convention as the analytic reduced-Hessian path.

    Returns
    -------
    np.ndarray of shape (n_params, n_params) or None
    """
    param_names = [name for name, p in builder._parameters.items() if not p.fixed]
    n_params = len(param_names)

    if n_params == 0:
        return None

    # Build the simulation model and compile its evaluator ONCE. The kinetic
    # parameters enter as discopt Parameters, so each perturbed re-solve only
    # mutates Parameter.value and reuses the XLA cache -- no JAX recompile per
    # finite-difference step.
    from kipetax.model_tools.discopt_compat import get_nlp_evaluator
    NLPEvaluator = get_nlp_evaluator()
    from kipetax.model_tools.discopt_model_builder import solve_local_nlp

    try:
        m, dae, param_map = builder.build_parametric_simulation_model()
        evaluator = NLPEvaluator(m)
    except Exception:
        return None

    def residuals_at(P_values):
        for name, val in P_values.items():
            if name in param_map and val is not None:
                param_map[name].value = float(val)
        try:
            result = solve_local_nlp(m, evaluator=evaluator)
        except Exception:
            return None
        if result.status not in ('optimal', 'feasible'):
            return None
        Z = builder.extract_results(result, dae, {}).get('Z')
        if Z is None:
            return None
        return _residuals_from_Z(builder, Z, variances)

    # Collect measurement residuals at optimal solution
    residuals_base = residuals_at(P_opt)
    if residuals_base is None or len(residuals_base) == 0:
        return None

    n_meas = len(residuals_base)
    J = np.zeros((n_meas, n_params))

    for i, pname in enumerate(param_names):
        p_val = P_opt[pname]
        eps = max(abs(p_val) * eps_rel, eps_abs)

        P_pert = dict(P_opt)
        P_pert[pname] = p_val + eps

        residuals_pert = residuals_at(P_pert)
        if residuals_pert is None:
            continue

        J[:, i] = (residuals_pert - residuals_base) / eps

    # Cov(P) = (J^T J)^-1 * scale.  With known measurement variances the
    # residuals are already whitened, so scale=1; otherwise estimate the noise
    # level from the residuals via the reduced chi-square.
    JtJ = J.T @ J
    if scale is None:
        scale = np.sum(residuals_base**2) / max(n_meas - n_params, 1)

    try:
        cov = np.linalg.inv(JtJ) * scale
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JtJ) * scale

    return cov


def _residuals_from_Z(builder, Z, variances):
    """Measurement residual vector from a simulated trajectory ``Z``.

    Concentrations contribute ``(c_sim - c_meas) / sqrt(sigma)``; spectral data
    contribute the profiled residual ``D - C(theta) S_ls^T`` (see
    :func:`_spectral_residuals`). Returns an ``np.ndarray`` or ``None``.
    """
    residuals = []
    c_data = builder._get_concentration_data()
    for comp_name, df in c_data.items():
        if comp_name not in Z.columns:
            continue
        t_arr = np.array(Z.index)
        for t_meas, row in df.iterrows():
            c_meas = float(row.iloc[0])
            if np.isnan(c_meas):
                continue
            # Find nearest time in Z
            nearest_idx = np.argmin(np.abs(t_arr - float(t_meas)))
            c_sim = float(Z.iloc[nearest_idx][comp_name])
            sigma = variances.get(comp_name, 1.0)
            if sigma is None or sigma <= 0:
                sigma = 1.0
            residuals.append((c_sim - c_meas) / np.sqrt(sigma))

    # Spectral residuals: D - C(theta) S^T, with the pure-component spectra
    # S profiled out by their closed-form least-squares optimum at this
    # theta. Re-fitting S per perturbation makes the finite-difference
    # Jacobian the correct separable (VARPRO) derivative for theta.
    sd = builder._spectral_data
    if sd is not None and sd.D is not None:
        spec = _spectral_residuals(builder, Z, sd, variances)
        if spec is None:
            return None
        residuals.extend(spec.tolist())

    return np.array(residuals) if residuals else None


def _spectral_residuals(builder, Z, sd, variances):
    """Profiled spectral residual vector ``vec((D - C(theta) S_ls^T) / sqrt(sigma))``.

    ``C(theta)`` is the absorbing-component trajectory at the spectral
    measurement times (from the simulated ``Z``); ``S_ls = (C^T C)^+ C^T D`` is
    the closed-form optimal spectra at this ``theta``. Returns ``None`` if the
    absorbing-component trajectories are unavailable.
    """
    if builder._absorbing_components is not None:
        abs_comps = builder._absorbing_components
    else:
        abs_comps = [n for n in builder._components
                     if builder._components[n].absorbing
                     and n not in builder._non_absorbing]
    if not abs_comps:
        return None

    times = sd.times
    wavelengths = sd.wavelengths
    t_arr = np.array(Z.index, dtype=float)

    # C(theta): (n_times, n_abs) at the spectral measurement times.
    Cm = np.zeros((len(times), len(abs_comps)))
    for ci, comp in enumerate(abs_comps):
        if comp not in Z.columns:
            return None
        col = Z[comp].values
        for ti, tm in enumerate(times):
            Cm[ti, ci] = col[np.argmin(np.abs(t_arr - float(tm)))]

    Dm = sd.D.loc[times, wavelengths].values        # (n_times, n_lam)
    S_ls = np.linalg.pinv(Cm.T @ Cm) @ (Cm.T @ Dm)  # (n_abs, n_lam)
    resid = Dm - Cm @ S_ls                          # (n_times, n_lam)

    sigma = variances.get('device', 1.0)
    if sigma is None or sigma <= 0:
        sigma = 1.0
    return (resid / np.sqrt(sigma)).ravel()
