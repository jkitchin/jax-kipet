---
name: kipetax
description: Expert guidance for kipetax, a Pyomo-free JAX/discopt rewrite of KIPET for kinetic and spectral parameter estimation. Covers the model classes (ReactionModel, ReactionSet, PopulationBalanceModel, MixtureModel), the shared spectral observation model, simulation and estimation, variable projection, non-additive interaction spectra, and the non-obvious gotchas (time origin, variance scaling, solver choice, label ambiguity). Use when building or debugging kinetic/spectral fits, population balances, or blend-property models with kipetax; when the prompt mentions kipetax, KIPET, ReactionModel, ReactionSet, Arrhenius fits across experiments, spectral estimation, Beer-Lambert D=C S^T, MCR, population balance, or blend viscosity from spectra.
license: MIT
compatibility: Requires Python 3.12+ and kipetax with discopt>=0.10 and pounce-solver>=0.14 (`pip install git+https://github.com/jkitchin/jax-kipet`); see the Setup section.
---

# kipetax

kipetax estimates parameters for time-dependent transformations from data. It
keeps the familiar KIPET `ReactionModel` API but replaces all Pyomo internals
with [`discopt`](https://github.com/jkitchin/discopt) collocation and the
[`pounce`](https://github.com/jkitchin/pounce) NLP solver, and runs the ODE
integration and NLP in float64 JAX (`JAX_ENABLE_X64=1` is set on import).

## Setup

```bash
pip install git+https://github.com/jkitchin/jax-kipet   # or, from a checkout: pip install -e ".[dev]"
python -c "import kipetax, discopt, pounce; print(discopt.__version__, pounce.__version__)"
```

kipetax needs Python 3.12+, `discopt[jax]>=0.10` and `pounce-solver>=0.14`
(imported as `pounce`). If a solve fails with an unexpected keyword or a missing
`discopt.solvers.nlp_pounce`, check those versions first. For writing raw
discopt models (not needed for kipetax), install the companion skill:
`uvx knowhow add jkitchin/discopt --skill discopt`.

## The one idea to hold onto: the shared observation model

Every model class shares one observation model. The measured signal is a sum of
contributions, each a coefficient that depends on the latent state times a
spectrum:

```
D(t, ·) = sum_j  g_j(state(t)) · phi_j(·)
```

Models differ in only three choices:
1. **coefficient `g_j`** — a bare species concentration, or an expression of the state (`c_A·c_B`), or a blend fraction.
2. **spectrum `phi_j`** — **estimated** from the data, or **known** fixed physics.
3. **state** — a finite species list, or a distribution over an internal coordinate.

Everything else is shared. Keep this table in mind when choosing a class:

| Class / mode | spectra are | estimated or known | coefficient |
|---|---|---|---|
| `ReactionModel`, concentration fit | none (measure concentrations) | — | identity |
| `ReactionModel`, spectral fit `D=C S^T` | pure-component spectra `S` | **estimated** with the kinetics | species conc. `c_k(t)` |
| `ReactionModel.interaction(...)` | non-additive band `Gamma` | **estimated**, may be signed | expression, e.g. `c_A·c_B` |
| `run_opt(method='varpro')` | same `S` | **projected out** `S*=C^+ D` | species conc. |
| `PopulationBalanceModel` | a **known kernel** (size ruler) | **known** physics | section population `N(r,t)·w` |
| `MixtureModel` | component fingerprints | **estimated** by calibration | blend fraction `w_i` (+ `w_i w_j`) |

## Which class to use

- **Finite species, kinetics from concentration or spectral data** -> `ReactionModel`.
- **Non-additive spectra** (adduct band `~c_A c_B`, environment peak shift) -> `ReactionModel` + `interaction()`.
- **Wide spectral axis** (hundreds+ of wavelengths) -> `ReactionModel` with `run_opt(method='varpro')`.
- **State is a distribution over an internal coordinate** (particle radius, chain length), observed through a known kernel -> `PopulationBalanceModel`.
- **Static blend-property prediction, no time axis** -> `MixtureModel`.

## ReactionModel

```python
from kipetax import ReactionModel

r = ReactionModel('ABC')
k1 = r.parameter('k1', value=1.0, bounds=(0.1, 10))     # estimated (not fixed)
k2 = r.parameter('k2', value=0.2, bounds=(0.01, 5))
A = r.component('A', value=1.0)                          # value = initial conc at t0
B = r.component('B', value=0.0)
C = r.component('C', value=0.0)
r.add_ode('A', -k1 * A)
r.add_ode('B',  k1 * A - k2 * B)
r.add_ode('C',  k2 * B)
r.add_data(data=df)                                     # columns bind by name to components/states
r.set_discretization(nfe=40, ncp=3)
r.run_opt(variances={'A': 1e-4, 'B': 1e-4, 'C': 1e-4})
ro = r.results_dict['p_estimator']
print(ro.P, ro.Z)                                       # params, trajectories
```

- Fixed parameters: `r.parameter('kd', value=0.5, fixed=True)` or `r.fix_parameter('kd')`.
- Non-concentration states: `r.state('T', value=...)`; constants: `r.constant('Yxs', value=...)`.
- `simulate()` needs `set_time(t_end)`; estimation infers the span from data.
- Math in ODE expressions: `from kipetax import exp, log, sqrt, sin, cos, tan, log10, abs_`.

### Spectral estimation (`D = C S^T`)

```python
r.add_data(name='spectral', data=D, category='spectral')  # D: rows=time, cols=wavelength
r.run_opt(variances={'device': 1e-4})
ro.S   # recovered pure-component spectra (wavelengths x absorbing components)
```

### Non-additive interaction spectra

An adduct band scales as `c_A·c_B`; a peak shift is a signed derivative
spectrum. Add it as a column of `C` whose value is an EXPRESSION of the states:

```python
r.interaction('AB', A * B)               # fitted Gamma_AB comes back as ro.S['AB']
r.set_spectral_regularization(ridge=1e-3)  # keep the augmented fit identified
```

Diagnose the NEED for an interaction term first (rank inflation, structured
residual: see `Ex_6_interaction_diagnostics.py`) before augmenting. Model
selection precedes model augmentation.

### Solver choice: collocation vs varpro

`run_opt(method=...)`:
- `'collocation'` (default) — transcribes the ODEs to one NLP, `S` are decision variables, solved by pounce. Enforces `S >= 0`, gives joint `S`/kinetics covariance, but cost grows ~QUADRATICALLY in the wavelength count.
- `'varpro'` — variable projection: `S` eliminated analytically (`S* = C^+ D`), so the optimizer sees only the kinetics and cost is INDEPENDENT of wavelength count. Integrates the ODE by differentiable sequential shooting. Use for a wide spectral axis.

### Fused spectral + concentration data

The two objectives add; they are not alternatives. A spectral fit is invariant
under `c -> a*c, S -> S/a`, so when an initial condition is itself unknown the
absolute scale is free. Attach a few direct concentration measurements (an
offline GC endpoint, one calibrated channel) with a second `add_data(...)` call
alongside the D-matrix to pin it; both blocks are weighted by `1/variance`.

### Uncertainty

`r.run_opt(variances=..., covariance=True)` adds `ro.confidence_intervals`
(95%). `r.run_variance_estimator(...)` estimates measurement variances first
(`results_dict['v_estimator']`) when you do not know the noise level.

## ReactionSet (several experiments, shared parameters)

```python
from kipetax import ReactionSet, exp

# per experiment (one ReactionModel each, own data), centered Arrhenius form:
kref = r.parameter('kref', value=1.0, bounds=(1e-3, 1e3))
Ea   = r.parameter('Ea',   value=60.0, bounds=(0.0, 250.0))
iT   = r.constant('iT', value=1.0 / T - 1.0 / T_ref)   # differs per experiment
k    = kref * exp(-Ea / R_GAS * iT)

rs = ReactionSet('arrhenius')
for r in models: rs.add_model(r)
rs.set_shared_parameter('kref', value=1.0, bounds=(1e-3, 1e3))
rs.set_shared_parameter('Ea', value=60.0, bounds=(0.0, 250.0))
ro = rs.run_opt(variances={'device': 1e-4, 'run_573K': {'device': 4e-4}})
ro.P                      # shared names, plus '<experiment>.<local>' for the rest
ro.experiments['run_573K'].S
```

- Each experiment keeps its OWN `S`; spectra are never tied across experiments.
- Prefer the centered `1/T - 1/T_ref` form: fitting `A` and `Ea` directly is
  badly correlated.
- `covariance=True` is not implemented for `ReactionSet` yet (it warns).

## PopulationBalanceModel

For a state that is a distribution over an internal coordinate, observed through
a KNOWN spectral kernel `A = (N·w) @ Kernel^T`. It does NOT use the collocation
NLP (a mass-balance-coupled PBE densifies it and a single collocation solve can
take minutes); it integrates SEQUENTIALLY with a differentiable fixed-step scheme
and fits by JAX autodiff + L-BFGS-B.

```python
import numpy as np, jax.numpy as jnp
from kipetax import PopulationBalanceModel

pbm = PopulationBalanceModel('growth')
pbm.set_coordinate('r', np.linspace(0.8, 6.0, 45))
pbm.parameter('kg', value=0.01, bounds=(1e-3, 0.1))     # fit in LOG space -> bounds > 0
pbm.set_initial_state(jnp.zeros(45))                    # distribution, then any scalars
pbm.set_dynamics(step_fn, t_end=40.0, nt=1600, scheme='euler')  # step_fn(state, t, p) -> dstate
pbm.set_observation(observe_fn)                         # observe_fn(history, p) -> (n_obs, n_lam)
pbm.add_data(A_data)                                    # (n_obs, n_lambda)
ro = pbm.run_opt()
ro.P['kg']; ro.distribution()                           # rate + reconstructed N(r)
```

- `scheme='euler'` for upwind advection (positivity), `'rk4'` for smooth stiff systems.
- Extra scalar states (e.g. lignin recondensation extent `Q`) go at the END of the flat state vector.
- `PopulationBalanceModel.nnls_inversion(spectra, kernel)` is the naive per-spectrum baseline the mechanism-constrained fit is meant to beat.

## MixtureModel

Static (no time axis) blend property: `property = sum_i w_i P_i + sum_{i<j} w_i w_j Gamma_ij`.
Linear in parameters given the fractions, so `fit` is least squares.

```python
from kipetax import MixtureModel

spec = MixtureModel(n_components=4).fit(W_train, D_train)   # spectra
visc = MixtureModel(n_components=4).fit(W_train, lnv_train) # ln viscosity
visc.interaction_parameter(0, 3)                           # Grunberg-Nissan d_ij (or Gamma_ij band)
w = spec.infer_composition(a_measured_spectrum)            # inverse: spectrum -> fractions
```

## Gotchas (hard-won; check these first when a fit misbehaves)

1. **Time origin.** A component's `value=` is its concentration at t0, and
   `set_time(span)` fixes t0=0. If data starts late (first measurement > 0),
   kipetax integrates from t=0 and WARNS. Pre-fix, inferring t0 from the first
   measurement silently solved a different model and returned a confidently wrong
   rate with `status='optimal'`. Set `set_time()` explicitly when in doubt.
2. **Variance scaling.** Residuals are weighted by `1/variance`. A variance below
   ~1e-7 (i.e. `1/variance` beyond ~1e7) makes the collocation objective enormous
   and pounce reports `UNBOUNDED` on a sum of squares; `run_opt` now RAISES with a
   scaling hint. Supply a realistic measurement variance.
3. **Failures are loud.** `simulate()` and `run_opt()` raise `RuntimeError` on a
   failed solve rather than returning empty results. `simulate(solver_opts={'max_iter': N})`
   caps a stiff feasibility solve.
4. **Stiff systems / fast transients.** A uniform collocation mesh can miss a fast
   initial layer and converge to a wrong constant with `status='optimal'`. Use a
   graded mesh: `set_discretization(ncp=3, element_boundaries=eb)` clustering
   elements where the solution moves.
5. **varpro label ambiguity.** With free spectra, varpro identifies only the SET
   of ODE eigenvalues, not their assignment (for A->B->C, k1<->k2 give identical
   SSE). A species non-negativity penalty resolves it toward the physical
   assignment; use `method='collocation'` if you need the hard `S >= 0` constraint.
6. **interaction() is ALGEBRAIC, not differential.** Do NOT encode `c_A c_B` as a
   pseudo-species with a product-rule ODE — that adds a spurious quadratic
   differential constraint and biases the fit. Use `r.interaction('AB', A*B)`.
7. **Everything solves through pounce**, never discopt's default global
   branch-and-bound. Do not call `discopt.Model.solve()` directly.
8. **Preprocessed spectra.** `S >= 0` is a statement about raw absorbance.
   After baseline or scatter correction (MSC, SNV) pure spectra genuinely go
   negative; pass `run_opt(spectra_nonneg=False)` or the false constraint
   biases the kinetics, not just `S`.

## Examples (in `kipetax/examples/`)

`Ex_1` simulation; `Ex_2` concentration estimation; `Ex_3` hybrid bioreactor
(Monod from Raman under baselines); `Ex_4` non-additive spectra + varpro demo;
`Ex_5` polymerization moments (MW distribution from conversion); `Ex_6`
interaction diagnostics (rank/residual); `Ex_7` model-free DMD + diffusion
coordinate; `Ex_8` QD growth (population balance from UV-Vis); `Ex_9` lignin
depolymerization; `Ex_10` asphalt blending (`MixtureModel`).

Run the test suite with `pytest -m "not slow"` for the fast checks.
