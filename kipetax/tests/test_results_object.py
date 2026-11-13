"""
Tests for ResultsObject.
"""
import numpy as np
import pandas as pd
import pytest

from kipetax.estimator_tools.results_object import ResultsObject


def test_results_object_init():
    """ResultsObject initializes with empty defaults."""
    ro = ResultsObject()
    assert ro.P == {}
    assert ro.Z.empty
    assert ro.X.empty
    assert ro.S is None
    assert ro.sigma_sq == {}
    assert ro.confidence_intervals == {}
    assert ro.status == 'unknown'
    assert ro.objective is None


def test_load_from_builder_result():
    """load_from_builder_result populates all fields."""
    t = np.linspace(0, 5, 20)
    Z = pd.DataFrame({'A': np.exp(-t), 'B': 1 - np.exp(-t)}, index=t)
    X = pd.DataFrame(index=t)

    result_dict = {'Z': Z, 'X': X, 'Y': pd.DataFrame(), 'P': {'k1': 2.5}, 'S': None}

    ro = ResultsObject()
    ro.load_from_builder_result(result_dict)

    assert ro.P == {'k1': 2.5}
    assert not ro.Z.empty
    assert 'A' in ro.Z.columns
    assert 'B' in ro.Z.columns


def test_compute_var_norm_Z():
    """compute_var_norm returns a positive float."""
    t = np.linspace(0, 5, 20)
    Z = pd.DataFrame({'A': np.exp(-t)}, index=t)
    ro = ResultsObject()
    ro.Z = Z
    norm = ro.compute_var_norm('Z', order=np.inf)
    assert norm > 0


def test_compute_var_norm_empty():
    """compute_var_norm returns 0 for empty DataFrame."""
    ro = ResultsObject()
    norm = ro.compute_var_norm('Z')
    assert norm == 0.0


def test_str_repr():
    """__str__ and __repr__ run without errors."""
    ro = ResultsObject()
    ro.P = {'k1': 1.23}
    ro.status = 'optimal'
    ro.objective = 0.001
    s = str(ro)
    r = repr(ro)
    assert 'k1' in s
    assert 'optimal' in s
    assert 'optimal' in r


def test_sigma_sq_and_ci():
    """sigma_sq and confidence_intervals are dicts."""
    ro = ResultsObject()
    ro.P = {'k1': 2.0}
    ro.sigma_sq = {'A': 1e-4}
    ro.confidence_intervals = {'k1': (1.9, 2.1)}

    s = str(ro)
    assert '1.9' in s or '2.1' in s


def test_identifiability_fields_default_none():
    """New identifiability fields default to None."""
    ro = ResultsObject()
    assert ro.covariance is None
    assert ro.fim is None
    assert ro.identifiability is None
    assert ro.cov_source is None


def test_load_identifiability_identifiable():
    """load_identifiability sets finite CIs for an identifiable parameter."""
    from kipetax.calculation_tools.reduced_hessian import assess_identifiability

    fim = np.array([[1600.0]])
    report = assess_identifiability(fim, ['k1'])
    cov = np.array([[1.0 / 1600.0]])
    cov_result = {
        'covariance': cov, 'fim': fim, 'report': report,
        'param_names': ['k1'], 'cov_source': 'reduced_hessian',
    }

    ro = ResultsObject()
    ro.P = {'k1': 2.0}
    ro.load_identifiability(cov_result)

    assert ro.cov_source == 'reduced_hessian'
    assert ro.identifiability is report
    lo, hi = ro.confidence_intervals['k1']
    assert np.isfinite(lo) and np.isfinite(hi) and lo < 2.0 < hi

    s = str(ro)
    assert 'FIM rank' in s
    assert '1/1' in s
    assert 'NOT identifiable' not in s


def test_load_identifiability_unidentifiable():
    """A rank-deficient FIM yields infinite CIs and a flagged parameter."""
    from kipetax.calculation_tools.reduced_hessian import assess_identifiability

    # Rank-1 FIM in two parameters: only the (1,1)-direction is constrained.
    g = np.array([1.5, 2.0])             # proportional sensitivity columns
    fim = np.outer(g, g) * 1000.0
    report = assess_identifiability(fim, ['k1', 'k2'])
    cov = np.array([[np.inf, 0.0], [0.0, np.inf]])
    cov_result = {
        'covariance': cov, 'fim': fim, 'report': report,
        'param_names': ['k1', 'k2'],
        'cov_source': 'reduced_hessian(rank-deficient)',
    }

    ro = ResultsObject()
    ro.P = {'k1': 2.0, 'k2': 1.5}
    ro.load_identifiability(cov_result)

    assert ro.confidence_intervals['k1'] == (float('-inf'), float('inf'))
    assert ro.confidence_intervals['k2'] == (float('-inf'), float('inf'))
    assert not report.is_fully_identifiable

    s = str(ro)
    assert 'RANK DEFICIENT' in s
    assert 'NOT identifiable' in s
    assert 'transient excitation' in s
