# kipetax

**Kinetic Parameter Estimation Toolkit** — a Pyomo-free rewrite of
[KIPET](https://github.com/kwmcbride/KIPET) built on
[`discopt`](https://pypi.org/project/discopt/) and JAX.

kipetax estimates kinetic parameters (rate constants and more) for chemical
reaction systems from experimental data. You describe a reaction with
components, parameters, and ODEs; supply measured concentration or
spectroscopic data; and kipetax solves the resulting dynamic-optimization
problem to recover the parameters that best fit the data.

It keeps the familiar `ReactionModel` API of the original KIPET while replacing
all Pyomo internals with `discopt`'s collocation-based DAE modeling and NLP
solving. ODE integration and the NLP run in float64 (JAX) for accuracy.

---

## Features

- **Simulation** — forward-integrate a kinetic model from ODEs and initial
  conditions.
- **Parameter estimation from concentration data** — nonlinear least-squares
  fitting of rate constants against measured concentrations.
- **Spectral parameter estimation** — fit parameters from a spectroscopic
  D-matrix (absorbance over time × wavelength) via the Beer–Lambert relation,
  recovering both kinetics and pure-component spectra (S matrix).
- **Non-additive spectra** — `interaction(name, expr)` adds a column to the
  Beer–Lambert model for a band that is not a fixed pure-component spectrum: an
  adduct band scaling as `c_A·c_B`, an environment-driven peak shift. The fitted
  interaction spectrum is returned as an extra column of `S`.
- **Fused spectral + concentration data** — the two objectives are additive, not
  alternatives. A spectral fit is invariant under `c -> a*c, S -> S/a`, so where
  an initial condition is itself unknown the absolute scale is free; attaching a
  few directly measured concentrations (an offline GC endpoint, one calibrated
  product channel) alongside the D-matrix pins it. Both blocks are weighted by
  `1/variance`, so they are commensurate.
- **Variance estimation** — estimate measurement noise variances from data.
- **Confidence intervals** — optional covariance / 95% CI computation for
  estimated parameters.
- **Multiple experiments** — `ReactionSet` performs combined estimation across
  several experiments with shared parameters, from concentration data, spectral
  D-matrices, or both. The canonical use is an Arrhenius fit: one isothermal
  experiment per temperature, sharing `A` and `Ea`.
- **States, algebraics, constants, volume** — beyond simple concentrations.
- **Population balances** — `PopulationBalanceModel` handles a state that is a
  *distribution* over an internal coordinate (particle radius, chain length)
  rather than a finite species list, observed through a known spectral kernel.
  It integrates the balance by differentiable sequential shooting and fits the
  kinetics by JAX autodiff, the right tool where collocation would densify.
- **Graded collocation mesh** — `set_discretization(element_boundaries=...)`
  clusters finite elements to resolve a fast initial transient.

## Installation

```bash
pip install -e .
# with dev/plotting extras:
pip install -e ".[dev]"
```

Requires Python ≥ 3.12 and depends on `numpy`, `scipy`, `pandas`,
`discopt[jax]` ≥ 0.10, and `pounce-solver` ≥ 0.14 (the POUNCE NLP solver).

> **Note:** kipetax sets `JAX_ENABLE_X64=1` on import to enable float64
> precision, which is required for accurate ODE integration and NLP solving.

### Agent skill

kipetax ships a [`SKILL.md`](kipetax/skills/kipetax/SKILL.md) that teaches a
coding agent (Claude Code, Codex, OpenCode, Gemini CLI, Cursor, ...) the model
classes, the estimation workflow and the gotchas. Install it with
[knowhow](https://github.com/jkitchin/knowhow) (or `npx skills`, which uses the
same layout):

```bash
uvx knowhow add jkitchin/jax-kipet --skill kipetax        # this project
uvx knowhow add jkitchin/jax-kipet --skill kipetax -g     # all projects
```

From a checkout, `bash kipetax/skills/install.sh` (`--user` for `~/.claude`)
copies it into a Claude Code skills directory. The companion
[`discopt`](https://github.com/jkitchin/discopt) skill installs the same way
(`uvx knowhow add jkitchin/discopt --skill discopt`).

## Quick start

### Simulation (A → B → C)

```python
from kipetax import ReactionModel

r1 = ReactionModel('ABC_simulation')

# Parameters (fixed for a pure simulation)
k1 = r1.parameter('k1', value=2.0, bounds=(0, 10), fixed=True)
k2 = r1.parameter('k2', value=0.5, bounds=(0, 5), fixed=True)

# Components with initial concentrations
A = r1.component('A', value=1.0)
B = r1.component('B', value=0.0)
C = r1.component('C', value=0.0)

# ODEs: A → B → C
r1.add_ode('A', -k1 * A)
r1.add_ode('B',  k1 * A - k2 * B)
r1.add_ode('C',  k2 * B)

r1.set_time(10.0)
r1.set_discretization(nfe=50, ncp=3)
r1.simulate()

ro = r1.results_dict['simulator']
print(ro.Z.tail())   # concentration trajectories (DataFrame: time × components)
```

### Parameter estimation from concentration data

```python
import pandas as pd
from kipetax import ReactionModel

r1 = ReactionModel('ABC_estimation')

# Parameters to estimate (note: not fixed; provide bounds + initial guess)
k1 = r1.parameter('k1', value=1.0, bounds=(0.1, 10))
k2 = r1.parameter('k2', value=0.2, bounds=(0.01, 5))

A = r1.component('A', value=1.0)
B = r1.component('B', value=0.0)
C = r1.component('C', value=0.0)

r1.add_ode('A', -k1 * A)
r1.add_ode('B',  k1 * A - k2 * B)
r1.add_ode('C',  k2 * B)

# Measured concentration data: DataFrame indexed by time, columns = components
r1.add_data(name='concentration_data', data=df)   # or file='data.csv'
r1.set_discretization(nfe=40, ncp=3)

r1.run_opt(variances={'A': 1e-4, 'B': 1e-4, 'C': 1e-4})

ro = r1.results_dict['p_estimator']
print(ro.P)          # estimated parameters {'k1': ..., 'k2': ...}
print(ro.objective)
```

### Spectral parameter estimation

```python
import kipetax

r1 = kipetax.ReactionModel('spectral-pe')
k1 = r1.parameter('k1', value=0.3, bounds=(0.0, 5.0))
k2 = r1.parameter('k2', value=1.4, bounds=(0.0, 3.0))
A = r1.component('A', value=1.0)
B = r1.component('B', value=0.0)
C = r1.component('C', value=0.0)
r1.add_ode('A', -k1 * A)
r1.add_ode('B',  k1 * A - k2 * B)
r1.add_ode('C',  k2 * B)

# D: DataFrame with rows = time, columns = wavelength, values = absorbance
r1.add_data(name='spectral', data=D, category='spectral')
r1.set_discretization(nfe=30, ncp=3)

r1.run_opt(variances={'device': 1e-4})

ro = r1.results_dict['p_estimator']
print(ro.P)          # estimated rate constants
print(ro.S)          # pure-component spectra (wavelengths × components)
```

`run_opt` takes a `method` argument. The default, `'collocation'`, transcribes
the ODEs to one NLP and makes the spectra `S` decision variables, so its cost
grows quadratically in the number of wavelengths but it enforces `S ≥ 0` and
gives joint confidence intervals. For a wide spectral axis, `method='varpro'`
uses **variable projection**: `S` is eliminated analytically (`S* = C⁺D`), so the
optimizer sees only the kinetic parameters and the cost is independent of the
wavelength count. It integrates the ODEs by differentiable sequential shooting.

```python
r1.run_opt(variances={'device': 1e-4}, method='varpro')   # scales to 1000s of wavelengths
```

Because varpro uses free spectra it identifies only the *set* of ODE eigenvalues;
a species non-negativity penalty resolves the assignment (e.g. which rate is
A→B vs B→C). Use `'collocation'` when you need the `S ≥ 0` constraint or joint
`S`/kinetics covariance; use `'varpro'` when the spectral axis is large.

### Multiple experiments with shared parameters

```python
from kipetax import ReactionModel, ReactionSet

# build r1, r2 as above (each with its own data) ...

rs = ReactionSet('combined')
rs.add_model(r1)
rs.add_model(r2)
rs.set_shared_parameter('k1', value=1.0, bounds=(0.1, 10))

ro = rs.run_opt(variances={'A': 1e-4, 'B': 1e-4, 'C': 1e-4})
print(ro.P)                      # shared estimates, plus '<experiment>.<local>'
print(ro.experiments['exp1'].S)  # each experiment's own trajectories and spectra
```

Each experiment contributes whatever data it carries — a concentration table, a
spectral D-matrix, or both. Every experiment keeps its **own** `S`: tying
pure-component spectra across experiments is only valid when the optical
configuration is identical, and band shapes drift with temperature, so it is
not assumed.

Parameters named in `set_shared_parameter` are estimated jointly; everything
else stays local to its experiment and is reported under `'<experiment>.<name>'`
so two experiments' local estimates cannot collide. Per-experiment measurement
noise is set with a nested override — different temperatures rarely share a
noise level:

```python
ro = rs.run_opt(variances={'device': 1e-4,                 # applies to all
                           'run_573K': {'device': 4e-4}})  # overrides one
```

An Arrhenius fit is the natural use: one isothermal experiment per temperature,
sharing the kinetic parameters. Give each experiment its inverse-temperature
offset as a `constant` — this centered form decorrelates the prefactor from the
activation energy far better than fitting `A` and `Ea` directly:

```python
kref = r.parameter('kref', value=1.0, bounds=(1e-3, 1e3))
Ea   = r.parameter('Ea',   value=60.0, bounds=(0.0, 250.0))
iT   = r.constant('iT', value=1.0 / T - 1.0 / T_ref)   # differs per experiment
k    = kref * exp(-Ea / R_GAS * iT)
```

`covariance=True` is not yet implemented for combined estimation and warns.

(role-of-spectra)=
## The role of spectra across the models

Every model in kipetax shares one observation model. The measured signal is a
sum of contributions, each a coefficient that depends on the model's latent
state times a spectrum:

```
D(t, ·) = Σ_j  g_j(state(t)) · φ_j(·)
```

The models differ in only three choices: whether the coefficient `g_j` is a bare
species concentration or an expression of the state, whether the spectrum `φ_j`
is **estimated** from the data or **known** physics, and whether the state is a
finite species list or a distribution over an internal coordinate. Everything
else is shared.

| Model / mode | What the spectra are | Estimated or known | Coefficient `g_j` |
|---|---|---|---|
| `ReactionModel`, concentration fit | none — concentrations measured directly | — | identity |
| `ReactionModel`, spectral fit (`D = C·Sᵀ`) | pure-component spectra `S` | **estimated** jointly with the kinetics | species concentration `c_k(t)` |
| `ReactionModel.interaction(...)` | non-additive band `Γ` (adduct, peak shift) | **estimated**, may be signed | an expression, e.g. `c_A·c_B` |
| `method='varpro'` | same pure-component spectra `S` | **projected out** (`S* = C⁺D`), not optimized | species concentration |
| `PopulationBalanceModel` | a **known kernel** (a size/conjugation ruler `λ_peak(r)`) | **known** fixed physics | section population `N(r,t)·w` |
| `MixtureModel` | per-component property fingerprints | **estimated** by calibration over composition | blend fraction `w_i` (and `w_i·w_j`) |

Read across the models, the spectra play the same role in each: they are the
forward map from the quantity you care about but cannot see directly to the
signal you measure.

- In a **spectral `ReactionModel` fit** the pure-component spectra are unknowns.
  Each absorbing species has one fixed spectrum, the absorbance is their
  concentration-weighted sum (Beer–Lambert bilinearity), and the fit recovers
  both the kinetics and the spectra `S`. This is the classical MCR / KIPET
  picture.
- **`interaction()`** relaxes the "one fixed spectrum per species" assumption.
  An adduct band scales as `c_A·c_B` and a peak shift is a signed derivative
  spectrum; both are extra columns whose spectra are still estimated but attach
  to an expression of the state rather than a single concentration.
- **Variable projection** (`method='varpro'`) keeps the very same spectral model
  but eliminates `S` analytically instead of optimizing it, so the cost no
  longer grows with the number of wavelengths. The spectra are still there; they
  are just computed in closed form at the end rather than carried as unknowns.
- In a **`PopulationBalanceModel`** the spectra are **not** unknowns. A particle
  of radius `r` has a known exciton peak `λ_peak(r)`, so the kernel is fixed
  physics — a size ruler. The measured spectrum is an integral of the unknown
  size distribution against that fixed kernel, and recovering the distribution
  from one spectrum is an ill-posed inversion that the mechanism regularizes.
  A non-additive band (lignin recondensation) is one more known-spectrum column.
- In a **`MixtureModel`** the spectra are per-component fingerprints recovered by
  calibration over a training set of blends, and the model runs both ways:
  forward (predict a blend's spectrum or viscosity) and inverse (infer
  composition from a measured spectrum).

## Core API

### `ReactionModel`

| Method | Purpose |
|---|---|
| `parameter(name, value=, bounds=, fixed=, variance=)` | Declare a kinetic parameter (estimated unless `fixed=True`). |
| `component(name, value=, bounds=, absorbing=)` | Declare a chemical species (concentration state). |
| `state(name, ...)` / `volume(...)` | Declare a non-concentration state (e.g. temperature, volume). |
| `constant(name, value=)` | Declare a fixed model constant. |
| `algebraic(name, ...)` / `add_reaction(...)` / `add_expression(...)` | Declare algebraic variables / named reaction terms. |
| `add_ode(var, expr)` / `add_odes(dict)` | Set ODE right-hand sides (`dx/dt = expr`). |
| `add_data(name=, file=, data=, category=)` | Attach concentration data, or spectral D-matrix with `category='spectral'`. |
| `set_time(t_end)` | Set the time horizon (t₀ = 0). Inferred from data if omitted during estimation. |
| `set_discretization(nfe=, ncp=, scheme=)` | Collocation settings: finite elements, collocation points, `'radau'`/`'legendre'`. |
| `simulate(parameters=None)` | Forward simulation → `results_dict['simulator']`. |
| `run_opt(variances=, covariance=False)` | Parameter estimation → `results_dict['p_estimator']`. |
| `run_variance_estimator(...)` | Estimate measurement variances → `results_dict['v_estimator']`. |
| `plot(var='Z')` | Quick matplotlib plot of `Z`/`X`/`Y`. |
| `diagnostics()` | Print a model summary. |

ODE expressions are written with the proxy objects returned by `parameter`,
`component`, etc. Math functions for use inside expressions are exported from
the package:

```python
from kipetax import exp, log, sqrt, sin, cos, tan, log10, abs_
```

### `ResultsObject`

Returned via `results_dict`. Key attributes:

- `P` — estimated parameters `{name: value}`
- `Z` — concentration trajectories (DataFrame, time × components)
- `X` — non-concentration state trajectories
- `Y` — algebraic-variable trajectories
- `S` — pure-component spectra (spectral problems only)
- `sigma_sq` — variance estimates
- `confidence_intervals` — 95% CIs (when `covariance=True`)
- `status`, `objective` — solver status and objective value
- `experiments` — `{name: ResultsObject}` per experiment (`ReactionSet` only)

### `ReactionSet`

Container for multiple `ReactionModel`s. Use `add_model`, declare shared
parameters with `set_shared_parameter`, then `run_opt` to estimate jointly.
`run_opt` returns a `ResultsObject` whose `experiments` attribute maps each
experiment name to its own `ResultsObject` (`Z`, `X`, `Y`, `S`, `P`); the same
object is stored on the member model's `results_dict['p_estimator']`, so a
member reads back exactly as it would after a single-model fit.

### `PopulationBalanceModel`

For transformations whose state is a *distribution* over an internal coordinate
(particle radius, chain length) rather than a finite species list. Where
`ReactionModel` transcribes to a collocation NLP and solves with POUNCE, a
population balance couples every section through mass balance and coagulation,
which densifies that NLP; so this class integrates the balance *sequentially*
with a differentiable fixed-step scheme (`jax.lax.scan`) and fits the kinetics
by JAX autodiff plus L-BFGS-B. The observation is bilinear with a **known**
kernel, `A = (N · w) @ Kernel.T`, the same object as a spectral fit but with the
spectra fixed rather than estimated.

```python
import numpy as np, jax.numpy as jnp
from kipetax import PopulationBalanceModel

pbm = PopulationBalanceModel('growth')
pbm.set_coordinate('r', np.linspace(0.8, 6.0, 45))
pbm.parameter('kg', value=0.01, bounds=(1e-3, 0.1))   # fit in log space
pbm.set_initial_state(jnp.zeros(45))
pbm.set_dynamics(step_fn, t_end=40.0, nt=1600, scheme='euler')  # step_fn(N, t, p)
pbm.set_observation(observe_fn)                        # observe_fn(history, p)
pbm.add_data(A_data)                                   # (n_obs, n_lambda)
ro = pbm.run_opt()
print(ro.P['kg'], ro.distribution())                  # rate + reconstructed N(r)
```

`PopulationBalanceModel.nnls_inversion(spectra, kernel)` provides the naive
per-spectrum baseline that the mechanism-constrained fit is meant to beat.

### `MixtureModel`

The interaction motif once more, with no time axis: a blend property is a sum of
component contributions plus pairwise `w_i·w_j` terms,
`property = Σ w_i P_i + Σ w_i w_j Γ_ij`. For a spectrum `Γ_ij` is an interaction
band; for viscosity the same form is the Grunberg–Nissan rule on `ln η`. Both are
linear in their parameters, so `fit` is a least-squares solve and `predict` a
matrix product. `infer_composition` runs the inverse (recover blend fractions
from a measured spectrum) as a simplex-constrained least squares.

```python
from kipetax import MixtureModel

spec = MixtureModel(n_components=4).fit(W_train, D_train)   # spectra
visc = MixtureModel(n_components=4).fit(W_train, lnv_train) # ln viscosity
d_03 = visc.interaction_parameter(0, 3)                     # Grunberg-Nissan d_ij
w = spec.infer_composition(a_measured_spectrum)             # spectrum -> fractions
```

## Examples & notebooks

- `kipetax/examples/Ex_1_ode_sim.py` — A → B → C simulation.
- `kipetax/examples/Ex_2_estimation.py` — parameter estimation on synthetic
  noisy concentration data, comparing recovered vs. true parameters.
- `kipetax/examples/Ex_3_bioreactor_monod.py` — hybrid Monod estimation from
  Raman spectra buried under fluorescence and turbidity baselines, resolved by
  2nd-derivative preprocessing plus the mechanism.
- `kipetax/examples/Ex_4_spectral_interactions.py` — non-additive Raman spectra:
  an additive fit is biased on interacting data; `interaction('AB', A*B)` removes
  the bias and recovers the adduct spectrum.
- `kipetax/examples/Ex_5_polymerization_moments.py` — free-radical
  polymerization by method of moments; recovers `kp` from conversion alone and
  reconstructs the molecular-weight distribution the probe never measures.
- `kipetax/examples/Ex_6_interaction_diagnostics.py` — the model-selection step:
  rank inflation and structured residuals that flag additivity failure *before*
  augmenting the model.
- `kipetax/examples/Ex_7_model_free_coordinates.py` — DMD/Koopman rates and a
  diffusion-map reaction coordinate with no mechanism assumed, and where linear
  DMD breaks on nonlinear kinetics.
- `kipetax/examples/Ex_8_qd_growth_uvvis.py` — quantum-dot size distribution
  from UV-Vis via `PopulationBalanceModel`; the mechanism regularizes an
  ill-posed Fredholm inversion.
- `kipetax/examples/Ex_9_lignin_depolymerization.py` — lignin depolymerization
  (chain-length population balance with a non-additive recondensation band).
- `kipetax/examples/Ex_10_asphalt_blending.py` — static blend-property model
  (`MixtureModel`): predict viscosity from a spectrum through the shared
  `w_i·w_j` interaction structure.
- `notebooks/01_simulation.ipynb` — simulation walkthrough.
- `notebooks/02_concentration_parameter_estimation.ipynb` — concentration-based
  estimation.
- `notebooks/03_spectral_parameter_estimation.ipynb` — spectral estimation,
  including D-matrix reconstruction (D ≈ C·Sᵀ).

Run an example:

```bash
python kipetax/examples/Ex_1_ode_sim.py
```

## Project layout

```
kipetax/
  main_modules/       ReactionModel, ReactionSet (top-level API)
  model_components/   element blocks, datasets, spectral data handler
  model_tools/        discopt model builder, proxy variables/expressions
  estimator_tools/    simulator, parameter & variance estimators, results
  calculation_tools/  interpolation, covariance, helpers
  variance_methods/   variance-estimation methods
  input_output/       data loading
  visuals/            plotting
  examples/           runnable example scripts
  tests/              pytest suite
notebooks/            example Jupyter notebooks
```

## Testing

```bash
pytest
```

Markers (see `pytest.ini`):

- `slow` — tests that run NLP solves (> 60 s each)
- `integration` — full end-to-end tests

```bash
pytest -m "not slow"     # skip the slow solves
```

## Documentation

API documentation is generated with [Sphinx](https://www.sphinx-doc.org) from
the in-source NumPy-style docstrings.

```bash
# build the HTML docs (uv handles the docs dependencies)
uv run --with-editable ".[docs]" sphinx-build -b html docs docs/_build/html

# or, inside an environment that has the docs extras installed:
make -C docs html
```

The rendered site lands in `docs/_build/html/index.html`.

## License

MIT. See [`LICENSE`](LICENSE).

kipetax is an independent reimplementation that follows the user-facing API of
[KIPET](https://github.com/kwmcbride/KIPET) (GPL-3.0) for compatibility; it
contains no KIPET source code. See [`NOTICE`](NOTICE) for attribution and the
KIPET papers to cite alongside kipetax.
```
