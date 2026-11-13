"""
Variable-projection (VARPRO) estimator for spectral kinetic fits.

The default spectral path in :class:`DiscoptModelBuilder` solves the SIMULTANEOUS
problem: the pure-component spectra ``S`` (n_abs x n_lambda) are NLP decision
variables alongside the collocated concentrations, and the bilinear objective
``||D - C S^T||^2`` couples them densely in the KKT system. Its cost grows
roughly QUADRATICALLY in the number of wavelengths, because ``S`` alone
contributes ``n_abs * n_lambda`` variables.

Variable projection removes ``S`` from the optimization entirely. For fixed
kinetic parameters ``theta`` the concentrations ``C(theta)`` are determined, and
the ``S`` that minimizes ``||D - C S^T||^2`` is the linear least-squares solution
``S* = C^+ D``. Substituting it back leaves a residual that depends only on
``theta``:

    r(theta) = (I - C C^+) D          (the projection of D off the column space of C)

so the optimizer sees only the two or three kinetic parameters, no matter how
many wavelengths there are. ``S`` is recovered once, at the end. This is the
classical Golub-Pereyra separable-least-squares method, and it is why the
standalone demo fits 400 wavelengths in seconds.

Because ``S`` is never a collocation variable, this estimator does not use the
collocation NLP at all. It integrates the ODEs SEQUENTIALLY with a differentiable
fixed-step RK4 (``jax.lax.scan``) and optimizes ``theta`` with L-BFGS-B, gradients
by ``jax.value_and_grad`` differentiating through both the integrator and the
projection. Interaction columns (:meth:`ReactionModel.interaction`) are handled
for free: they are extra columns of ``C``, projected out with the species.
"""
import numpy as np

from kipetax.estimator_tools.results_object import ResultsObject
from kipetax.model_tools.proxy_vars import resolve_expression


def _jnp_functions():
    import jax.numpy as jnp
    return {'exp': jnp.exp, 'log': jnp.log, 'log10': lambda x: jnp.log10(x),
            'sqrt': jnp.sqrt, 'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
            'abs': jnp.abs}


def _resolve_nonneg_weight(spectra_nonneg, default):
    """Turn the public ``spectra_nonneg`` argument into a penalty weight.

    ``True`` -> the default weight, ``False`` -> 0.0, a number -> itself.
    """
    if spectra_nonneg is True:
        return float(default)
    if spectra_nonneg is False:
        return 0.0
    weight = float(spectra_nonneg)
    if weight < 0:
        raise ValueError("spectra_nonneg must be True, False, or a "
                         f"non-negative weight; got {spectra_nonneg!r}")
    return weight


class VarProEstimator:
    """Separable-least-squares spectral estimation by variable projection.

    Parameters
    ----------
    builder : DiscoptModelBuilder
        A populated builder (components, parameters, ODEs, spectral data).
    """

    # Default weight on the species-spectra non-negativity penalty. With free
    # spectra, VARPRO identifies only the SET of ODE eigenvalues, not their
    # assignment to reactions: for A->B->C the column space
    # span{exp(-k1 t), exp(-k2 t), 1} is symmetric in k1<->k2, so the projected
    # residual is identical for either label and the optimizer keeps whichever
    # ordering it started from. The simultaneous collocation path breaks this
    # tie with its S >= 0 constraint; here a penalty on negative SPECIES spectra
    # does the same. It is scale-invariant (S and the residual both scale
    # linearly with D).
    #
    # It does NOT bias a solution whose spectra really are non-negative -- which
    # is true of raw absorbance and FALSE after the standard chemometric
    # front-end. MSC applies a per-spectrum affine map, which is not
    # Beer-Lambert-preserving, so the pure-component spectra of
    # baseline-corrected data carry genuinely negative regions (10%-ish of the
    # band energy of a minor component is typical). On such data this default is
    # strong enough to distort the kinetics badly, and the right setting is
    # ``spectra_nonneg=False`` or a weight an order of magnitude smaller. See
    # ``ReactionModel.run_opt``.
    _NONNEG_WEIGHT = 2.0

    def __init__(self, builder):
        self._b = builder

    # -- model extraction --

    def _ode_vars(self):
        """State variables with an ODE, in a stable order (components then states)."""
        b = self._b
        names = [n for n in b._components if n in b._odes_dict]
        names += [n for n in b._states if n in b._odes_dict]
        return names

    def _initial_state(self):
        b = self._b
        y0 = []
        for n in self._ode_vars():
            elem = b._components.get(n) or b._states.get(n)
            y0.append(float(elem.value) if elem.value is not None else 0.0)
        return np.array(y0, dtype=float)

    def _fitted_and_fixed_params(self):
        b = self._b
        fitted, fixed = {}, {}
        for name, p in b._parameters.items():
            if getattr(p, 'fixed', False):
                fixed[name] = float(p.value)
            else:
                lo, hi = self._b._get_param_bounds(p)
                fitted[name] = {'value': float(p.value), 'bounds': (lo, hi)}
        return fitted, fixed

    # -- integration --

    def _build_forward(self, ode_vars, fixed, const_vals):
        """Return ``concentrations(theta_dict, t_query)`` as a JAX function.

        Integrates the ODEs by fixed-step RK4 on a uniform grid over the data
        span, then linearly interpolates each trajectory to the measurement
        times.
        """
        import jax
        import jax.numpy as jnp

        odes = self._b._odes_dict
        funcs = _jnp_functions()
        y0 = jnp.asarray(self._initial_state())
        t0, tf = self._b._times

        # Fixed integration grid: fine enough to resolve the kinetics, capped for
        # compile cost. Measurement times are interpolated onto it.
        n_steps = max(200, 4 * self._b._nfe)
        dt = (tf - t0) / n_steps
        grid = t0 + jnp.arange(n_steps + 1) * dt

        def rhs(y, theta):
            subst = {n: y[i] for i, n in enumerate(ode_vars)}
            subst.update(const_vals)
            subst.update(fixed)
            subst.update(theta)
            return jnp.stack([jnp.asarray(
                resolve_expression(odes[n], subst, functions=funcs), dtype=jnp.float64)
                for n in ode_vars])

        def integrate(theta):
            def stepper(y, _):
                k1 = rhs(y, theta)
                k2 = rhs(y + 0.5 * dt * k1, theta)
                k3 = rhs(y + 0.5 * dt * k2, theta)
                k4 = rhs(y + dt * k3, theta)
                yn = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                return yn, yn
            _, hist = jax.lax.scan(stepper, y0, None, length=n_steps)
            return jnp.concatenate([y0[None, :], hist], axis=0)   # (n_steps+1, n_vars)

        def concentrations(theta, t_query):
            hist = integrate(theta)
            # Column-wise linear interpolation onto the query times.
            cols = [jnp.interp(t_query, grid, hist[:, i])
                    for i in range(len(ode_vars))]
            return jnp.stack(cols, axis=1), integrate(theta)      # (n_query, n_vars)

        return concentrations, grid

    def _design_columns(self, C_query, ode_vars):
        """Assemble the C design matrix: absorbing species + interaction columns."""
        import jax.numpy as jnp

        idx = {n: i for i, n in enumerate(ode_vars)}
        abs_names = self._b._absorbing_names()
        cols = [C_query[:, idx[n]] for n in abs_names if n in idx]

        int_names = list(self._b._interactions)
        funcs = _jnp_functions()
        if int_names:
            const_vals = {n: (c.value if c.value is not None else 0.0)
                          for n, c in self._b._constants.items()}
            subst = {n: C_query[:, idx[n]] for n in ode_vars}
            subst.update(const_vals)
            for nm in int_names:
                cols.append(jnp.asarray(resolve_expression(
                    self._b._interactions[nm], subst, functions=funcs)))
        return jnp.stack(cols, axis=1), abs_names, int_names

    def _concentration_targets(self, ode_vars, variances):
        """Direct concentration measurements, fused with the spectral residual.

        A spectral fit is invariant under ``c -> a*c, S -> S/a``, so the absolute
        scale of every trajectory is set only by the initial conditions. Where
        those are themselves unknown (an unknown site density, a coverage of
        unknown absolute magnitude) the scale is free and any rate constant that
        multiplies a concentration inherits the ambiguity. A few directly
        measured concentrations -- an offline GC endpoint, one calibrated product
        channel -- pin it.

        Returns
        -------
        list of (state_index, times, values, sigma)
        """
        idx = {n: i for i, n in enumerate(ode_vars)}
        targets = []
        for name, df in self._b._get_concentration_data().items():
            if name not in idx:
                continue
            sub = df.dropna()
            if sub.empty:
                continue
            sigma = (variances or {}).get(name, 1.0)
            if sigma is None or sigma <= 0:
                sigma = 1.0
            targets.append((idx[name],
                            np.asarray(sub.index, dtype=float),
                            np.asarray(sub.iloc[:, 0].values, dtype=float),
                            float(sigma)))
        return targets

    # -- estimation --

    def run_opt(self, variances=None, covariance=False, spectra_nonneg=True):
        """Fit the kinetic parameters by variable projection.

        Parameters
        ----------
        spectra_nonneg : bool or float, optional
            Strength of the non-negativity penalty on the species spectra.
            ``True`` (default) uses ``_NONNEG_WEIGHT``, ``False`` disables it,
            and a float sets the weight directly. See ``ReactionModel.run_opt``.

        Returns
        -------
        ResultsObject
        """
        import jax
        import jax.numpy as jnp
        from scipy.optimize import minimize

        b = self._b
        sd = b._spectral_data
        if sd is None or sd.D is None:
            raise ValueError("VARPRO requires spectral data")

        ode_vars = self._ode_vars()
        fitted, fixed = self._fitted_and_fixed_params()
        if not fitted:
            raise ValueError("no free parameters to estimate")
        const_vals = {n: (c.value if c.value is not None else 0.0)
                      for n, c in b._constants.items()}

        times = np.asarray(sd.times, dtype=float)
        wavelengths = sd.wavelengths
        D = np.asarray(sd.D.loc[sd.times, wavelengths].values, dtype=float)
        finite = np.isfinite(D)
        D_filled = np.where(finite, D, 0.0)
        Dj = jnp.asarray(D_filled)
        maskj = jnp.asarray(finite.astype(float))

        concentrations, _grid = self._build_forward(ode_vars, fixed, const_vals)
        t_query = jnp.asarray(times)

        # Concentration measurements are appended to the spectral query times so
        # that one integration serves both residuals.
        conc_targets = self._concentration_targets(ode_vars, variances)
        n_spec = len(times)
        if conc_targets:
            t_all = jnp.concatenate(
                [t_query] + [jnp.asarray(t) for _, t, _, _ in conc_targets])
            conc_blocks = [(i, len(t), jnp.asarray(y), s)
                           for i, t, y, s in conc_targets]
        else:
            t_all = t_query
            conc_blocks = []

        sigma_device = (variances or {}).get('device', 1.0)
        if sigma_device is None or sigma_device <= 0:
            sigma_device = 1.0
        sigma_device = float(sigma_device)

        names = list(fitted)
        log_lo = np.array([np.log(fitted[n]['bounds'][0]) for n in names])
        log_hi = np.array([np.log(fitted[n]['bounds'][1]) for n in names])
        theta0 = np.clip(np.array([np.log(fitted[n]['value']) for n in names]),
                         log_lo, log_hi)

        n_abs = len(b._absorbing_names())
        nonneg_weight = _resolve_nonneg_weight(spectra_nonneg,
                                               self._NONNEG_WEIGHT)

        def predict_and_project(theta_log):
            theta = {n: jnp.exp(theta_log[i]) for i, n in enumerate(names)}
            C_all, _hist = concentrations(theta, t_all)
            C_query = C_all[:n_spec]
            C, _abs, _int = self._design_columns(C_query, ode_vars)
            # S* = C^+ D, then residual R = C S* - D  (masked for missing data).
            S = jnp.linalg.lstsq(C, Dj, rcond=None)[0]
            R = (C @ S - Dj) * maskj
            return R, S, C, C_all[n_spec:]

        def loss(theta_log):
            R, S, _C, C_conc = predict_and_project(theta_log)
            # Non-negativity penalty on the SPECIES spectra only (rows 0..n_abs);
            # interaction spectra may be signed (a band shift is a derivative).
            neg = jnp.minimum(S[:n_abs], 0.0)
            # Dividing the spectral block (residual AND penalty together) by the
            # device variance keeps the penalty's documented scale invariance
            # while making the block commensurate with the 1/sigma-weighted
            # concentration terms. With no concentration data and the default
            # unit variance this is exactly the unweighted objective.
            obj = (jnp.sum(R * R)
                   + nonneg_weight * jnp.sum(neg * neg)) / sigma_device
            off = 0
            for state_idx, n_pts, y_meas, sigma_c in conc_blocks:
                resid = C_conc[off:off + n_pts, state_idx] - y_meas
                obj = obj + jnp.sum(resid * resid) / sigma_c
                off += n_pts
            return obj

        vg = jax.jit(jax.value_and_grad(loss))

        def scipy_obj(theta_log):
            v, g = vg(jnp.asarray(theta_log))
            return float(v), np.asarray(g, dtype=float)

        res = minimize(scipy_obj, theta0, jac=True, method='L-BFGS-B',
                       bounds=list(zip(log_lo, log_hi)),
                       options=dict(ftol=1e-14, gtol=1e-10, maxiter=500))

        theta_opt = res.x
        grad = np.asarray(res.jac, dtype=float)
        at_lo = np.isclose(theta_opt, log_lo) & (grad > 0)
        at_hi = np.isclose(theta_opt, log_hi) & (grad < 0)
        proj = np.where(at_lo | at_hi, 0.0, grad)
        converged = res.success or np.linalg.norm(proj) < 1e-4

        P = {n: float(np.exp(theta_opt[i])) for i, n in enumerate(names)}
        P.update(fixed)

        ro = ResultsObject()
        ro.P = P
        ro.status = 'optimal' if converged else 'failed'
        ro.objective = float(res.fun)

        # Recover S and the full trajectories at the optimum.
        R_opt, S_opt, C_opt, _C_conc = predict_and_project(jnp.asarray(theta_opt))
        S_np = np.asarray(S_opt)                     # (n_abs+n_int, n_lam)
        _, abs_names, int_names = self._design_columns(
            concentrations({n: jnp.exp(theta_opt[i]) for i, n in enumerate(names)},
                           t_query)[0], ode_vars)
        import pandas as pd
        s_cols = list(abs_names) + list(int_names)
        ro.S = pd.DataFrame(S_np.T, index=wavelengths, columns=s_cols)
        ro.Z, ro.X = self._full_trajectories(theta_opt, names, ode_vars)

        if covariance:
            self._covariance(ro, names, theta_opt, predict_and_project,
                             variances, finite.sum())
        return ro

    def _full_trajectories(self, theta_opt, names, ode_vars):
        """Dense Z (components) and X (states) trajectories at the optimum."""
        import jax.numpy as jnp
        import pandas as pd

        concentrations, grid = self._build_forward(
            ode_vars, self._fitted_and_fixed_params()[1],
            {n: (c.value if c.value is not None else 0.0)
             for n, c in self._b._constants.items()})
        theta = {n: jnp.exp(theta_opt[i]) for i, n in enumerate(names)}
        _, hist = concentrations(theta, jnp.asarray(grid))
        hist = np.asarray(hist)
        t = np.asarray(grid)

        comp_names = [n for n in self._b._components if n in self._b._odes_dict]
        state_names = [n for n in self._b._states if n in self._b._odes_dict]
        idx = {n: i for i, n in enumerate(ode_vars)}
        Z = pd.DataFrame({n: hist[:, idx[n]] for n in comp_names}, index=t)
        X = (pd.DataFrame({n: hist[:, idx[n]] for n in state_names}, index=t)
             if state_names else pd.DataFrame(index=t))
        return Z, X

    def _covariance(self, ro, names, theta_opt, predict_and_project, variances,
                    n_meas):
        """Gauss-Newton covariance of the kinetic parameters from the projected
        residual Jacobian (the Golub-Pereyra Jacobian; jax differentiates through
        the lstsq automatically)."""
        import jax
        import jax.numpy as jnp
        try:
            def resid_natural(theta_nat):
                theta_log = jnp.log(theta_nat)
                R, _S, _C, _Cc = predict_and_project(theta_log)
                return R.ravel()

            theta_nat = np.exp(theta_opt)
            J = np.asarray(jax.jacobian(resid_natural)(jnp.asarray(theta_nat)))
            # Scale: known variance -> 1, else reduced chi-square SSE/(m - p).
            sigma = variances.get('device') if variances else None
            if sigma and sigma > 0:
                scale = float(sigma)
            else:
                dof = max(int(n_meas) - len(names), 1)
                scale = float(ro.objective) / dof
            JTJ = J.T @ J
            cov = scale * np.linalg.pinv(JTJ)
            ro.covariance = cov
            ro.cov_source = 'varpro_gauss_newton'
            from scipy.stats import norm
            z = float(norm.ppf(0.975))
            for i, n in enumerate(names):
                std = np.sqrt(max(cov[i, i], 0.0))
                ro.confidence_intervals[n] = (ro.P[n] - z * std, ro.P[n] + z * std)
        except Exception as e:
            print(f"Warning: VARPRO covariance failed ({e}).")
