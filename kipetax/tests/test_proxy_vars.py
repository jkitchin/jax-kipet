"""
Tests for the proxy variable system.
"""
import pytest
from kipetax.model_tools.proxy_vars import (
    KipetProxy, PendingBinaryOp, PendingUnaryOp, PendingFunctionCall,
    exp, log, sqrt, resolve_expression
)


def test_basic_proxy_creation():
    k1 = KipetProxy('k1', 'parameter')
    assert k1.name == 'k1'
    assert k1.kind == 'parameter'


def test_binary_op_mul():
    k1 = KipetProxy('k1', 'parameter')
    A = KipetProxy('A', 'component')
    expr = k1 * A
    assert isinstance(expr, PendingBinaryOp)
    assert expr.op == '*'
    assert expr.left is k1
    assert expr.right is A


def test_binary_op_neg():
    k1 = KipetProxy('k1', 'parameter')
    A = KipetProxy('A', 'component')
    expr = -k1 * A
    assert isinstance(expr, PendingBinaryOp)
    assert expr.op == '*'
    left = expr.left
    assert isinstance(left, PendingUnaryOp)
    assert left.op == 'neg'


def test_binary_op_add():
    k1 = KipetProxy('k1', 'parameter')
    A = KipetProxy('A', 'component')
    B = KipetProxy('B', 'component')
    expr = k1 * A + B
    assert isinstance(expr, PendingBinaryOp)
    assert expr.op == '+'


def test_binary_op_sub():
    A = KipetProxy('A', 'component')
    B = KipetProxy('B', 'component')
    expr = A - B
    assert isinstance(expr, PendingBinaryOp)
    assert expr.op == '-'


def test_binary_op_div():
    A = KipetProxy('A', 'component')
    B = KipetProxy('B', 'component')
    expr = A / B
    assert isinstance(expr, PendingBinaryOp)
    assert expr.op == '/'


def test_binary_op_pow():
    A = KipetProxy('A', 'component')
    expr = A ** 2
    assert isinstance(expr, PendingBinaryOp)
    assert expr.op == '**'


def test_function_call_exp():
    k = KipetProxy('k', 'parameter')
    expr = exp(k)
    assert isinstance(expr, PendingFunctionCall)
    assert expr.func_name == 'exp'
    assert len(expr.args) == 1


def test_function_call_log():
    A = KipetProxy('A', 'component')
    expr = log(A)
    assert isinstance(expr, PendingFunctionCall)
    assert expr.func_name == 'log'


def test_function_call_sqrt():
    A = KipetProxy('A', 'component')
    expr = sqrt(A)
    assert isinstance(expr, PendingFunctionCall)
    assert expr.func_name == 'sqrt'


def test_nested_expression():
    """Arrhenius: k * exp(-Ea / (R * T))"""
    k = KipetProxy('k', 'parameter')
    Ea = KipetProxy('Ea', 'constant')
    R = KipetProxy('R', 'constant')
    T = KipetProxy('T', 'state')
    expr = k * exp(-Ea / (R * T))
    assert isinstance(expr, PendingBinaryOp)
    assert expr.op == '*'
    assert isinstance(expr.right, PendingFunctionCall)
    assert expr.right.func_name == 'exp'


def test_resolve_expression_float():
    """Resolving a plain float returns the float."""
    result = resolve_expression(3.14, {})
    assert result == pytest.approx(3.14)


def test_resolve_expression_proxy():
    k1 = KipetProxy('k1', 'parameter')
    subst = {'k1': 2.5}
    result = resolve_expression(k1, subst)
    assert result == pytest.approx(2.5)


def test_resolve_expression_binary():
    k1 = KipetProxy('k1', 'parameter')
    A = KipetProxy('A', 'component')
    expr = k1 * A
    subst = {'k1': 2.0, 'A': 3.0}
    result = resolve_expression(expr, subst)
    assert result == pytest.approx(6.0)


def test_resolve_expression_neg():
    k1 = KipetProxy('k1', 'parameter')
    A = KipetProxy('A', 'component')
    expr = -k1 * A
    subst = {'k1': 2.0, 'A': 3.0}
    result = resolve_expression(expr, subst)
    assert result == pytest.approx(-6.0)


def test_resolve_expression_nested():
    k1 = KipetProxy('k1', 'parameter')
    A = KipetProxy('A', 'component')
    B = KipetProxy('B', 'component')
    expr = -k1 * A + k1 * B
    subst = {'k1': 2.0, 'A': 3.0, 'B': 1.0}
    result = resolve_expression(expr, subst)
    assert result == pytest.approx(-4.0)


def test_resolve_expression_function():
    import math
    k = KipetProxy('k', 'parameter')
    expr = exp(k)
    subst = {'k': 1.0}
    result = resolve_expression(expr, subst)
    assert result == pytest.approx(math.e, rel=1e-6)


def test_resolve_missing_var_raises():
    k1 = KipetProxy('k1', 'parameter')
    with pytest.raises(KeyError):
        resolve_expression(k1, {})


def test_scalar_arithmetic():
    A = KipetProxy('A', 'component')
    expr = 2.0 * A + 1.0
    subst = {'A': 3.0}
    result = resolve_expression(expr, subst)
    assert result == pytest.approx(7.0)


def test_radd():
    A = KipetProxy('A', 'component')
    expr = 1.0 + A
    subst = {'A': 4.0}
    result = resolve_expression(expr, subst)
    assert result == pytest.approx(5.0)
