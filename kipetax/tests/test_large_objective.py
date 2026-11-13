"""
Dense datasets must not build a degenerate (deep) objective tree.

The least-squares objective has one term per measurement. Accumulating them
one at a time built an expression tree as deep as the dataset is long, and
discopt walks expression trees recursively, so past ~1000 measurements the
solve died with RecursionError. Inline spectroscopy or a logged HPLC easily
produces that many points.

These tests check the tree depth of the built objective directly instead of
solving: a 1200-point fit takes many minutes, because JAX tracing of the
per-measurement objective grows with the number of terms.
"""
import numpy as np
import pandas as pd
import pytest

import kipetax.model_tools.discopt_model_builder as builder_module
from kipetax import ReactionModel, ReactionSet
from kipetax.model_tools.discopt_model_builder import _balanced_sum

N_TIMES = 600          # x 2 measured species = 1200 objective terms
MAX_DEPTH = 60         # log2(1200) ~ 11, plus each residual's own small subtree


def _depth(expr):
    """Longest path through ``left``/``right`` links, computed without recursion."""
    deepest, stack = 0, [(expr, 1)]
    while stack:
        node, d = stack.pop()
        deepest = max(deepest, d)
        for child in (getattr(node, 'left', None), getattr(node, 'right', None)):
            if child is not None:
                stack.append((child, d + 1))
    return deepest


def _dense_ab(name, A0):
    t = np.linspace(0.0, 5.0, N_TIMES)
    A = A0 * np.exp(-0.7 * t)
    r = ReactionModel(name)
    k = r.parameter('k', value=0.3, bounds=(0.01, 5.0))
    a = r.component('A', value=A0)
    r.component('B', value=0.0)
    r.add_ode('A', -k * a)
    r.add_ode('B', k * a)
    r.add_data(data=pd.DataFrame({'A': A, 'B': A0 - A}, index=t))
    r.set_discretization(nfe=20, ncp=3)
    return r


def test_balanced_sum_is_exact():
    terms = list(range(1, 1001))
    assert _balanced_sum(terms) == sum(terms)
    assert _balanced_sum([5]) == 5
    with pytest.raises(ValueError):
        _balanced_sum([])


def test_dense_single_experiment_objective_is_shallow():
    r = _dense_ab('dense', 1.0)
    m, _, _ = r._make_builder().build_parameter_estimation_model(
        {'A': 1e-6, 'B': 1e-6})
    assert _depth(m._objective.expression) < MAX_DEPTH


class _Captured(Exception):
    pass


def test_dense_reaction_set_objective_is_shallow(monkeypatch):
    captured = {}

    def capture(m, *args, **kwargs):
        captured['model'] = m
        raise _Captured

    monkeypatch.setattr(builder_module, 'solve_local_nlp', capture)
    rs = ReactionSet('dense_set')
    rs.add_model(_dense_ab('e1', 1.0))
    rs.add_model(_dense_ab('e2', 2.0))
    with pytest.raises(_Captured):
        rs.run_opt(variances={'A': 1e-6, 'B': 1e-6})
    assert _depth(captured['model']._objective.expression) < MAX_DEPTH
