"""
Parameter covariance from POUNCE's reduced Hessian (the sIPOPT analog).

The analytic path in :mod:`kipetax.calculation_tools.reduced_hessian` builds a
*Gauss-Newton* Fisher Information Matrix from discopt's JAX constraint Jacobian.
POUNCE can instead hand back the **full reduced Hessian of the Lagrangian** at
the converged KKT point -- the complete second-order object, which additionally
accounts for active bounds and (for spectral fits) Schur-complements the free
pure-component spectra ``S`` out automatically, with no VARPRO special-casing.

Method (sIPOPT "pinned-constraint" reduced Hessian)
---------------------------------------------------
Pin each estimated parameter with an equality constraint ``P_k == theta_hat_k``;
POUNCE returns ``H = obj_scal * B K^-1 B^T`` over those pin rows -- the reduced
Hessian in *parameter coordinates*, reusing the converged KKT factorization.

POUNCE returns ``H`` in natural (unscaled) units (pounce#128). For a least
squares objective ``chi^2 = sum resid^2 / sigma`` the objective Hessian is
``2 * FIM``, so passing ``obj_scal = 0.5`` makes ``-H`` the Fisher information
and ``Cov = -inv(H)``. We form ``fim = -H`` and reuse
:func:`~kipetax.calculation_tools.reduced_hessian.assess_identifiability` /
:func:`~kipetax.calculation_tools.reduced_hessian.covariance_from_fim` so the
null-space / identifiability reporting is identical to the analytic path.
"""
from __future__ import annotations

import numpy as np

from kipetax.calculation_tools.reduced_hessian import (
    assess_identifiability, covariance_from_fim, _variable_offsets)

# obj_scal that turns the least-squares objective Hessian (= 2 * FIM) into the
# Fisher information, so that -H == FIM and Cov = -inv(H).
_LSQ_OBJ_SCAL = 0.5


def pounce_covariance(builder, model, dae, result, variances,
                      rank_tol=1e-8, scale=1.0, free_params=None):
    """Analytic covariance + identifiability from POUNCE's reduced Hessian.

    Parameters mirror
    :func:`kipetax.calculation_tools.reduced_hessian.analytic_covariance`.

    Returns
    -------
    dict with keys ``covariance``, ``fim``, ``report`` (IdentifiabilityReport),
    ``param_names``, and ``cov_source`` (``'pounce_reduced_hessian'`` or
    ``'pounce_reduced_hessian(rank-deficient)'``).
    """
    import pounce
    from kipetax.model_tools.discopt_compat import (get_extract_x_flat,
                                                     get_nlp_evaluator)
    NLPEvaluator = get_nlp_evaluator()
    extract_x_flat = get_extract_x_flat()
    from discopt.solvers.nlp_pounce import _IpoptCallbacks, _infer_constraint_bounds

    if free_params is None:
        free_params = [name for name, p in builder._parameters.items() if not p.fixed]
    n_params = len(free_params)
    if n_params == 0:
        return None

    # Estimated values at the solution, read straight off the solved primal.
    offsets = _variable_offsets(model)
    x_solved = np.asarray(extract_x_flat(result, model), dtype=np.float64)
    theta_hat = {name: float(x_solved[offsets[f"P_{name}"]]) for name in free_params}

    # Rebuild the estimation model and pin each parameter with an equality
    # constraint P_k == theta_hat_k (appended last, so the pins are the final
    # constraint rows). POUNCE's reduced Hessian over these rows is the reduced
    # Hessian in parameter coordinates.
    m, _dae, dm_params = builder.build_parameter_estimation_model(variances)
    for name in free_params:
        m.subject_to(dm_params[name] == theta_hat[name], name=f"_cov_pin_{name}")

    # Gauss-Newton objective Hessian (discopt#98): drops the objective's
    # residual-curvature second-order term, which matters for a bilinear
    # spectral objective (C @ S). NOTE this does NOT make the reduced Hessian
    # equal the Fisher information: the reduced Hessian still carries the
    # multiplier-weighted *constraint* curvature (the nonlinear DAE), which the
    # analytic FIM (first-order sensitivities only) omits. That term adds
    # spurious finite curvature to structurally unidentifiable directions, so
    # this backend can still MASK unidentifiability -- the analytic path is the
    # default for that reason.
    ev = NLPEvaluator(m, gauss_newton=True)
    n_con = ev.n_constraints
    pin_idx = list(range(n_con - n_params, n_con))

    # Warm-start at the converged primal (same variable layout; pins add no
    # variables) so the sensitivity solve lands on the reported solution.
    x0 = np.asarray(extract_x_flat(result, m), dtype=np.float64)

    lb, ub = ev.variable_bounds
    cl, cu = _infer_constraint_bounds(ev)
    problem = pounce.Problem(n=ev.n_variables, m=ev.n_constraints,
                             problem_obj=_IpoptCallbacks(ev),
                             lb=lb, ub=ub, cl=cl, cu=cu)
    problem.add_option("print_level", 0)

    _x, info = problem.solve_with_sens(
        x0, pin_idx, compute_reduced_hessian=True, obj_scal=_LSQ_OBJ_SCAL)

    rh = info.get("reduced_hessian")
    if rh is None:
        raise RuntimeError("POUNCE did not return a reduced Hessian "
                           "(solve may not have converged).")

    # Column-major flat (n_params^2) -> (n_params, n_params). With obj_scal=0.5
    # the Fisher information is -H; symmetrize to kill round-off asymmetry.
    H = np.asarray(rh, dtype=np.float64).reshape(n_params, n_params, order="F")
    fim = -0.5 * (H + H.T)

    report = assess_identifiability(fim, free_params, rank_tol=rank_tol)
    cov = covariance_from_fim(fim, report, scale=scale)
    source = "pounce_reduced_hessian" if report.is_fully_identifiable \
        else "pounce_reduced_hessian(rank-deficient)"
    return {
        "covariance": cov,
        "fim": fim,
        "report": report,
        "param_names": free_params,
        "cov_source": source,
    }
