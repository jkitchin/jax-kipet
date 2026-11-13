"""
VarianceEstimator for kipetax.

Simplified variance estimation: runs parameter estimation with unit variances,
then estimates component variances from the residuals.
"""
import numpy as np
import pandas as pd

from kipetax.estimator_tools.parameter_estimator import ParameterEstimator
from kipetax.estimator_tools.results_object import ResultsObject


class VarianceEstimator:
    """
    Estimates component variances from residuals after parameter estimation.

    Parameters
    ----------
    builder : DiscoptModelBuilder
    """

    def __init__(self, builder):
        self._builder = builder

    def run(self, tol=1e-4, max_iter=20, **kwargs):
        """
        Run variance estimation.

        Iterates:
        1. Estimate parameters with current variances
        2. Update variances from residuals
        Until convergence.

        Parameters
        ----------
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.

        Returns
        -------
        dict : {component_name: estimated_variance}
        """
        # Initialize variances to 1.0
        comp_names = list(self._builder._components.keys())
        variances = {name: 1.0 for name in comp_names}

        pe = ParameterEstimator(self._builder)

        for it in range(max_iter):
            ro = pe.run_opt(variances=variances)
            if ro.status not in ('optimal', 'feasible'):
                print(f"Variance estimation: solver failed at iteration {it}")
                break

            # Compute residuals per component
            c_data = self._builder._get_concentration_data()
            new_variances = {}
            for comp_name in comp_names:
                if comp_name not in c_data:
                    new_variances[comp_name] = variances[comp_name]
                    continue

                df = c_data[comp_name]
                Z = ro.Z

                residuals = []
                for t_meas, row in df.iterrows():
                    c_meas = float(row.iloc[0])
                    if np.isnan(c_meas):
                        continue
                    if comp_name in Z.columns:
                        t_arr = np.array(Z.index)
                        nearest = np.argmin(np.abs(t_arr - float(t_meas)))
                        c_sim = float(Z.iloc[nearest][comp_name])
                        residuals.append(c_sim - c_meas)

                if residuals:
                    new_variances[comp_name] = max(float(np.var(residuals)), 1e-12)
                else:
                    new_variances[comp_name] = variances[comp_name]

            # Check convergence
            max_change = max(abs(new_variances[k] - variances[k]) / max(variances[k], 1e-12)
                             for k in comp_names)
            variances = new_variances

            print(f"Variance estimation iter {it}: max_change={max_change:.4e}")
            if max_change < tol:
                print(f"Converged after {it+1} iterations")
                break

        return variances
