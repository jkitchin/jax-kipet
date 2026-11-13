"""
PopulationBalanceModel for kipetax.

A second model class alongside :class:`ReactionModel`, for transformations whose
state is a *distribution* over an internal coordinate rather than a finite list
of species: a particle-size distribution N(r, t), a chain-length distribution
N(i, t). These are population balances, and they are observed through a spectral
kernel that maps the coordinate to a wavelength (a quantum-confinement size
ruler for UV-Vis, a conjugation-length ruler for lignin fragments).

Why this is a separate class, and separately solved
---------------------------------------------------
``ReactionModel`` transcribes its ODEs to a single sparse NLP by orthogonal
collocation and solves it with POUNCE. That is the right tool for a handful of
kinetic states. It is the wrong tool for a population balance: the monomer mass
balance couples every section to every other (``m = M0 - c * sum_i N_i r_i^3``),
which densifies the KKT system, and a coagulation term convolves the
distribution with itself. Measured directly, a single collocation *simulation*
of a 25-section growth PBE with mass-balance coupling takes minutes; a fit needs
dozens of forward solves.

So this class keeps the physics differentiable but integrates it *sequentially*:
a fixed-step explicit scheme (upwind Euler or RK4) run with :func:`jax.lax.scan`,
which is bounded in cost, differentiable by plain reverse-mode through the scan,
and cannot stall on stiffness the way an adaptive integrator can. The kinetic
parameters are fit by ``L-BFGS-B`` on the log parameters, with gradients from
``jax.value_and_grad`` differentiating through both the integrator and the
observation. This is exactly the sequential-shooting route that ``ReactionModel``
contrasts itself against, and for a distributed state it is the practical one.

The observation is bilinear with a KNOWN kernel: ``A = (N . w) @ Kernel.T``. That
is ``D = C S^T`` with the size sections playing the role of species and ``S``
fixed to the kernel rather than estimated. A non-additive band that tracks a
scalar extent (lignin recondensation) is one more known-spectrum column. So the
observation is the same object the spectral ``ReactionModel`` fits; only the
number of concentrations is large and their spectra are known.

Everything the user supplies is a plain JAX function, so the class stays a thin,
honest harness: it owns the integrator, the fit loop, and result bookkeeping,
not the model.
"""
import numpy as np


class PopulationBalanceResults:
    """Container for a population-balance fit or simulation.

    Attributes
    ----------
    P : dict
        Estimated parameter values ``{name: value}``.
    state_history : ndarray
        Full integrated state, shape ``(nt + 1, state_dim)``.
    coordinate : ndarray or None
        The internal-coordinate grid, if one was set.
    status : str
        ``'optimal'``, ``'failed'``, or ``'simulated'``.
    objective : float or None
        Final least-squares objective (fits only).
    n_iter : int or None
        Optimizer iterations (fits only).
    """

    def __init__(self):
        self.P = {}
        self.state_history = None
        self.coordinate = None
        self.status = None
        self.objective = None
        self.n_iter = None
        self.message = None

    def distribution(self, step=-1):
        """The distribution part of the state at a step (default: final)."""
        if self.state_history is None:
            return None
        return np.asarray(self.state_history[step])


class PopulationBalanceModel:
    """A population balance over an internal coordinate, fit by sequential
    shooting.

    The workflow mirrors :class:`ReactionModel` but the moving parts are JAX
    functions rather than proxy expressions:

    >>> import jax.numpy as jnp, numpy as np
    >>> pbm = PopulationBalanceModel('growth')
    >>> pbm.set_coordinate('r', np.linspace(0.8, 6.0, 45))
    >>> pbm.parameter('kg', value=0.01, bounds=(1e-3, 0.1))
    >>> pbm.constant('M0', 1.0)
    >>> pbm.set_initial_state(jnp.zeros(45))
    >>> pbm.set_dynamics(step_fn, t_end=40.0, nt=1600, scheme='euler')
    >>> pbm.set_observation(observe_fn)
    >>> pbm.add_data(A_data)             # (n_obs, n_lambda)
    >>> res = pbm.run_opt()
    >>> res.P['kg']

    ``step_fn(state, t, p)`` returns the time derivative of the flat state
    vector; ``p`` is a dict of all parameter and constant values.
    ``observe_fn(state_history, p)`` returns predicted spectra
    ``(n_obs, n_lambda)`` aligned with :meth:`add_data`.
    """

    def __init__(self, name='pbm'):
        self.name = name
        self._coord_name = None
        self._coord = None
        self._params = {}       # name -> {'value', 'bounds'}
        self._constants = {}    # name -> value
        self._y0 = None
        self._rhs = None
        self._scheme = 'rk4'
        self._nt = 1000
        self._t_end = None
        self._observe = None
        self._data = None       # (n_obs, n_lambda)

    # -- specification --

    def set_coordinate(self, name, grid):
        """Set the internal coordinate (radius, chain length) and its grid."""
        self._coord_name = name
        self._coord = np.asarray(grid, dtype=float)
        return self._coord

    @property
    def coordinate(self):
        return self._coord

    def parameter(self, name, value, bounds=(1e-8, 1e8)):
        """Declare a parameter to estimate. Fit in log space, so bounds must be
        strictly positive."""
        lo, hi = bounds
        if lo <= 0:
            raise ValueError(f"parameter '{name}' is fit in log space; the "
                             f"lower bound must be > 0 (got {lo}).")
        self._params[name] = {'value': float(value), 'bounds': (float(lo), float(hi))}
        return name

    def constant(self, name, value):
        """Declare a fixed value available to the dynamics/observation."""
        self._constants[name] = value
        return name

    def set_initial_state(self, y0):
        """Set the initial flat state vector (distribution, then any scalars)."""
        self._y0 = y0

    def set_dynamics(self, rhs, t_end, nt, scheme='rk4'):
        """Set the RHS and integrator.

        Parameters
        ----------
        rhs : callable
            ``rhs(state, t, p) -> dstate/dt``, a JAX function.
        t_end : float
            Final time (t0 = 0).
        nt : int
            Number of fixed steps.
        scheme : {'rk4', 'euler'}
            Explicit integrator. Use 'euler' for upwind advection (positivity),
            'rk4' for smooth stiff systems.
        """
        if scheme not in ('rk4', 'euler'):
            raise ValueError("scheme must be 'rk4' or 'euler'")
        self._rhs = rhs
        self._t_end = float(t_end)
        self._nt = int(nt)
        self._scheme = scheme

    def set_observation(self, observe):
        """Set the observation operator ``observe(state_history, p) -> spectra``.

        ``state_history`` has shape ``(nt + 1, state_dim)``. The returned array
        must align row-for-row with :meth:`add_data`.
        """
        self._observe = observe

    def add_data(self, spectra):
        """Attach measured spectra, shape ``(n_obs, n_lambda)``."""
        self._data = np.asarray(spectra, dtype=float)

    # -- integration --

    def _integrator(self):
        """Build a jitted ``params_dict -> state_history`` integrator."""
        import jax
        import jax.numpy as jnp

        rhs = self._rhs
        y0 = jnp.asarray(self._y0)
        nt = self._nt
        dt = self._t_end / nt
        scheme = self._scheme

        def integrate(p_all):
            def _step(y, t):
                if scheme == 'euler':
                    yn = y + dt * rhs(y, t, p_all)
                else:
                    k1 = rhs(y, t, p_all)
                    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt, p_all)
                    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt, p_all)
                    k4 = rhs(y + dt * k3, t + dt, p_all)
                    yn = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                return yn, yn

            ts = jnp.arange(nt) * dt
            _, hist = jax.lax.scan(_step, y0, ts)
            return jnp.concatenate([y0[None, :], hist], axis=0)

        return jax.jit(integrate), self._constants

    def _merge_params(self, values):
        """Combine fitted values with constants into the dict passed to rhs."""
        p = dict(self._constants)
        p.update(values)
        return p

    def simulate(self, parameters=None):
        """Forward-integrate at the given (or initial) parameters.

        Returns
        -------
        PopulationBalanceResults
        """
        if self._rhs is None or self._y0 is None:
            raise ValueError("set_dynamics() and set_initial_state() first")
        vals = {n: self._params[n]['value'] for n in self._params}
        if parameters:
            vals.update(parameters)
        integrate, _ = self._integrator()
        hist = np.asarray(integrate(self._merge_params(vals)))

        ro = PopulationBalanceResults()
        ro.state_history = hist
        ro.coordinate = self._coord
        ro.P = dict(vals)
        ro.status = 'simulated'
        return ro

    # -- estimation --

    def run_opt(self, verbose=False):
        """Fit the parameters to the data by sequential-shooting least squares.

        Minimizes ``sum((observe(simulate(theta)) - data)^2)`` over the log
        parameters with L-BFGS-B, gradients by JAX autodiff through the
        integrator and observation.

        Returns
        -------
        PopulationBalanceResults
        """
        import jax
        import jax.numpy as jnp
        from scipy.optimize import minimize

        if self._rhs is None or self._observe is None or self._data is None:
            raise ValueError("set_dynamics(), set_observation(), and add_data() "
                             "are all required before run_opt()")
        if not self._params:
            raise ValueError("no parameters to estimate")

        names = list(self._params)
        log_lo = np.array([np.log(self._params[n]['bounds'][0]) for n in names])
        log_hi = np.array([np.log(self._params[n]['bounds'][1]) for n in names])
        theta0 = np.array([np.log(self._params[n]['value']) for n in names])
        theta0 = np.clip(theta0, log_lo, log_hi)

        integrate, _ = self._integrator()
        observe = self._observe
        consts = self._constants
        data = jnp.asarray(self._data)

        def loss(theta):
            vals = {n: jnp.exp(theta[i]) for i, n in enumerate(names)}
            p = {**consts, **vals}
            hist = integrate(p)
            pred = observe(hist, p)
            return jnp.sum((pred - data) ** 2)

        vg = jax.jit(jax.value_and_grad(loss))

        def scipy_obj(theta):
            v, g = vg(jnp.asarray(theta))
            return float(v), np.asarray(g, dtype=float)

        res = minimize(scipy_obj, theta0, jac=True, method='L-BFGS-B',
                       bounds=list(zip(log_lo, log_hi)),
                       options=dict(ftol=1e-14, gtol=1e-10,
                                    maxiter=500, disp=verbose))

        # Classify by the first-order optimality condition, not res.success.
        # With tolerances this tight, L-BFGS-B routinely stops "ABNORMAL" in the
        # line search right at a flat optimum -- res.success is False while the
        # projected gradient is ~1e-6. A small projected gradient at a
        # bound-feasible point IS a local optimum. Report the optimizer's own
        # message either way so a genuine failure is still visible.
        theta = np.asarray(res.x, dtype=float)
        grad = np.asarray(res.jac, dtype=float)
        # Free directions only: a component pinned to a bound and pushing into
        # it is stationary regardless of its gradient.
        at_lo = np.isclose(theta, log_lo) & (grad > 0)
        at_hi = np.isclose(theta, log_hi) & (grad < 0)
        proj_grad = np.where(at_lo | at_hi, 0.0, grad)
        converged = res.success or np.linalg.norm(proj_grad) < 1e-4

        vals = {n: float(np.exp(res.x[i])) for i, n in enumerate(names)}
        ro = PopulationBalanceResults()
        ro.P = vals
        ro.state_history = np.asarray(integrate(self._merge_params(vals)))
        ro.coordinate = self._coord
        ro.status = 'optimal' if converged else 'failed'
        ro.objective = float(res.fun)
        ro.n_iter = int(res.nit)
        ro.message = str(res.message)
        if verbose:
            print(f"PopulationBalance fit: status={ro.status} "
                  f"objective={ro.objective:.4e} iters={ro.n_iter} "
                  f"|proj grad|={np.linalg.norm(proj_grad):.2e}")
            for n in names:
                print(f"  {n:12s} = {vals[n]:.6g}")
        if not converged:
            print(f"Warning: population-balance fit did not converge "
                  f"({ro.message}); |proj grad|={np.linalg.norm(proj_grad):.2e}. "
                  f"Estimates are the last iterate and may be unreliable.")
        return ro

    @staticmethod
    def nnls_inversion(spectra, kernel):
        """Per-spectrum non-negative inversion onto a known kernel.

        The naive baseline: invert each spectrum independently onto the kernel
        with no mechanism tying the time points together. On an ill-posed
        Fredholm inversion this is exactly what the mechanism-constrained fit is
        meant to beat.

        Parameters
        ----------
        spectra : ndarray, shape (n_obs, n_lambda)
        kernel : ndarray, shape (n_lambda, n_sections)
            The forward map from sections to wavelengths.

        Returns
        -------
        ndarray, shape (n_obs, n_sections)
        """
        from scipy.optimize import nnls
        spectra = np.asarray(spectra, dtype=float)
        kernel = np.asarray(kernel, dtype=float)
        return np.array([nnls(kernel, np.maximum(a, 0.0))[0] for a in spectra])
