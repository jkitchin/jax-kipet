# Notebooks

Executable Jupyter notebooks live in the `notebooks/` directory. Each is a
self-contained walkthrough that runs top to bottom against synthetic data and
carries its saved outputs. The `Ex_N_*` notebooks mirror the scripts in
`kipetax/examples/` one-to-one.

Run any of them from the `notebooks/` directory:

```bash
cd notebooks
jupyter lab Ex_3_bioreactor_monod.ipynb
```

## Tutorial series

Hand-written, step-by-step introductions to the core API.

- {download}`01_simulation <../notebooks/01_simulation.ipynb>` — simulate the
  A → B → C reaction with `ReactionModel`.
- {download}`02_concentration_parameter_estimation <../notebooks/02_concentration_parameter_estimation.ipynb>`
  — estimate rate constants from noisy concentration data.
- {download}`03_spectral_parameter_estimation <../notebooks/03_spectral_parameter_estimation.ipynb>`
  — estimate from spectral (Beer-Lambert) data, including D-matrix
  reconstruction (D ≈ C·Sᵀ).
- {download}`04_rate_law_discovery_with_jaxsr <../notebooks/04_rate_law_discovery_with_jaxsr.ipynb>`
  — discover unknown rate laws from spectra alone, with unknown pure-component
  spectra: MCR-ALS curve resolution, [jaxsr](https://github.com/jkitchin/jaxsr)
  symbolic regression on the resolved profiles, then a kipetax refit on the raw
  spectra of several experiments with shared spectra, pruning and trajectory-BIC
  model selection. Requires `pip install jaxsr`.

## Worked examples

Ports of the `kipetax/examples/` demos, one notebook per script.

- {download}`Ex_3_bioreactor_monod <../notebooks/Ex_3_bioreactor_monod.ipynb>`
  — hybrid Monod estimation from Raman spectra buried under fluorescence and
  turbidity baselines, resolved by 2nd-derivative preprocessing plus the
  mechanism.
- {download}`Ex_4_spectral_interactions <../notebooks/Ex_4_spectral_interactions.ipynb>`
  — non-additive Raman spectra: an additive fit is biased on interacting data;
  `interaction('AB', A*B)` removes the bias and recovers the adduct spectrum.
- {download}`Ex_5_polymerization_moments <../notebooks/Ex_5_polymerization_moments.ipynb>`
  — free-radical polymerization by method of moments; recovers `kp` from
  conversion alone and reconstructs the molecular-weight distribution.
- {download}`Ex_6_interaction_diagnostics <../notebooks/Ex_6_interaction_diagnostics.ipynb>`
  — model selection: rank inflation and structured residuals that flag
  additivity failure before augmenting the model.
- {download}`Ex_7_model_free_coordinates <../notebooks/Ex_7_model_free_coordinates.ipynb>`
  — DMD/Koopman rates and a diffusion-map reaction coordinate with no mechanism
  assumed, and where linear DMD breaks on nonlinear kinetics.
- {download}`Ex_8_qd_growth_uvvis <../notebooks/Ex_8_qd_growth_uvvis.ipynb>`
  — quantum-dot size distribution from UV-Vis via `PopulationBalanceModel`; the
  mechanism regularizes an ill-posed Fredholm inversion.
- {download}`Ex_9_lignin_depolymerization <../notebooks/Ex_9_lignin_depolymerization.ipynb>`
  — lignin depolymerization (chain-length population balance with a
  non-additive recondensation band).
- {download}`Ex_10_asphalt_blending <../notebooks/Ex_10_asphalt_blending.ipynb>`
  — static blend-property model (`MixtureModel`): predict viscosity from a
  spectrum through the shared `w_i·w_j` interaction structure.
