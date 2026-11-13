"""
DiscoptSimulator for kipetax.

Replaces PyomoSimulator. Builds a simulation discopt model and extracts
the time-series trajectory.
"""
import numpy as np
import pandas as pd

from kipetax.estimator_tools.results_object import ResultsObject


class DiscoptSimulator:
    """
    Runs a pure ODE/DAE simulation using discopt collocation.

    Parameters
    ----------
    builder : DiscoptModelBuilder
        Populated builder object.
    """

    def __init__(self, builder):
        self._builder = builder

    def run_sim(self, parameters=None, solver_options=None):
        """
        Run the simulation.

        Parameters
        ----------
        parameters : dict, optional
            Override parameter values {name: value} for this simulation.
        solver_options : dict, optional
            Extra keyword arguments passed to the POUNCE solver.

        Returns
        -------
        ResultsObject
        """
        # Override parameter values if given
        if parameters is not None:
            for name, val in parameters.items():
                if name in self._builder._parameters:
                    self._builder._parameters[name].value = val

        m, dae = self._builder.build_simulation_model()

        solve_kwargs = {}
        if solver_options:
            solve_kwargs.update(solver_options)

        # Always solve through POUNCE (never discopt's branch-and-bound).
        result = self._builder.solve_local(m, solver_options=solve_kwargs)

        ro = ResultsObject()
        if result.status in ('optimal', 'feasible'):
            extracted = self._builder.extract_results(result, dae, {})
            ro.load_from_builder_result(extracted, result)
        else:
            ro.status = result.status
            ro.objective = result.objective

        return ro
