"""
Trajectory interpolation utilities for kipetax (pure numpy/scipy).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interpolate_trajectory(df, t_new, kind='linear', axis=0):
    """
    Interpolate a DataFrame trajectory to new time points.

    Parameters
    ----------
    df : DataFrame
        Time-indexed DataFrame (rows = times, columns = variables).
    t_new : array-like
        New time points.
    kind : str
        Interpolation type ('linear', 'cubic', etc.).
    axis : int
        Axis along which to interpolate (0 = rows / time axis).

    Returns
    -------
    DataFrame
        Interpolated values at t_new.
    """
    t_orig = np.array(df.index, dtype=float)
    t_new = np.array(t_new, dtype=float)

    result = {}
    for col in df.columns:
        f = interp1d(t_orig, df[col].values, kind=kind,
                     bounds_error=False, fill_value='extrapolate')
        result[col] = f(t_new)

    return pd.DataFrame(result, index=t_new)
