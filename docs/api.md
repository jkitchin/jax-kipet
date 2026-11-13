# API reference

## Model classes

The user-facing model classes. Each pairs a description of the transformation
with an observation model that links the latent quantity (concentrations, a
distribution, blend fractions) to the measured signal. See
{ref}`the role of spectra <role-of-spectra>` for how the observation models
relate.

```{eval-rst}
.. autoclass:: kipetax.main_modules.reaction_model.ReactionModel
   :members:

.. autoclass:: kipetax.main_modules.reaction_set.ReactionSet
   :members:

.. autoclass:: kipetax.main_modules.population_balance.PopulationBalanceModel
   :members:

.. autoclass:: kipetax.main_modules.population_balance.PopulationBalanceResults
   :members:

.. autoclass:: kipetax.main_modules.mixture_model.MixtureModel
   :members:
```

## Results

```{eval-rst}
.. autoclass:: kipetax.estimator_tools.results_object.ResultsObject
   :members:
```

## Estimators and simulator

```{eval-rst}
.. automodule:: kipetax.estimator_tools.discopt_simulator
   :members:

.. automodule:: kipetax.estimator_tools.parameter_estimator
   :members:

.. automodule:: kipetax.estimator_tools.varpro_estimator
   :members:

.. automodule:: kipetax.estimator_tools.variance_estimator
   :members:
```

## Model components

```{eval-rst}
.. automodule:: kipetax.model_components.model_components
   :members:

.. automodule:: kipetax.model_components.element_blocks
   :members:

.. automodule:: kipetax.model_components.data_component
   :members:

.. automodule:: kipetax.model_components.spectral_handler
   :members:
```

## Model tools

```{eval-rst}
.. automodule:: kipetax.model_tools.discopt_model_builder
   :members:

.. automodule:: kipetax.model_tools.proxy_vars
   :members:
```

## Calculation tools

```{eval-rst}
.. automodule:: kipetax.calculation_tools.interpolation
   :members:

.. automodule:: kipetax.calculation_tools.covariance
   :members:

.. automodule:: kipetax.calculation_tools.helper
   :members:
```
