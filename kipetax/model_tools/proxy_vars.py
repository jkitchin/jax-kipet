"""
Proxy Variable System for kipetax.

Users write ODE right-hand sides using KipetProxy objects, which build
pending expression trees. At solve time these trees are resolved into
real discopt expressions via resolve_expression().
"""
import math


class KipetProxy:
    """Lightweight node representing a named model variable in a pending expression tree."""

    def __init__(self, name, kind):
        self.name = name
        self.kind = kind  # 'parameter', 'component', 'state', 'algebraic', 'constant'

    def __repr__(self):
        return f"KipetProxy({self.name!r}, {self.kind!r})"

    def __neg__(self):
        return PendingUnaryOp('neg', self)

    def __add__(self, other):
        return PendingBinaryOp('+', self, other)

    def __radd__(self, other):
        return PendingBinaryOp('+', other, self)

    def __sub__(self, other):
        return PendingBinaryOp('-', self, other)

    def __rsub__(self, other):
        return PendingBinaryOp('-', other, self)

    def __mul__(self, other):
        return PendingBinaryOp('*', self, other)

    def __rmul__(self, other):
        return PendingBinaryOp('*', other, self)

    def __truediv__(self, other):
        return PendingBinaryOp('/', self, other)

    def __rtruediv__(self, other):
        return PendingBinaryOp('/', other, self)

    def __pow__(self, other):
        return PendingBinaryOp('**', self, other)

    def __rpow__(self, other):
        return PendingBinaryOp('**', other, self)

    def __abs__(self):
        return PendingFunctionCall('abs', self)


class PendingBinaryOp:
    """A binary operation node in a pending expression tree."""

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"PendingBinaryOp({self.op!r}, {self.left!r}, {self.right!r})"

    def __neg__(self):
        return PendingUnaryOp('neg', self)

    def __add__(self, other):
        return PendingBinaryOp('+', self, other)

    def __radd__(self, other):
        return PendingBinaryOp('+', other, self)

    def __sub__(self, other):
        return PendingBinaryOp('-', self, other)

    def __rsub__(self, other):
        return PendingBinaryOp('-', other, self)

    def __mul__(self, other):
        return PendingBinaryOp('*', self, other)

    def __rmul__(self, other):
        return PendingBinaryOp('*', other, self)

    def __truediv__(self, other):
        return PendingBinaryOp('/', self, other)

    def __rtruediv__(self, other):
        return PendingBinaryOp('/', other, self)

    def __pow__(self, other):
        return PendingBinaryOp('**', self, other)

    def __rpow__(self, other):
        return PendingBinaryOp('**', other, self)


class PendingUnaryOp:
    """A unary operation node in a pending expression tree."""

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"PendingUnaryOp({self.op!r}, {self.operand!r})"

    def __neg__(self):
        return PendingUnaryOp('neg', self)

    def __add__(self, other):
        return PendingBinaryOp('+', self, other)

    def __radd__(self, other):
        return PendingBinaryOp('+', other, self)

    def __sub__(self, other):
        return PendingBinaryOp('-', self, other)

    def __rsub__(self, other):
        return PendingBinaryOp('-', other, self)

    def __mul__(self, other):
        return PendingBinaryOp('*', self, other)

    def __rmul__(self, other):
        return PendingBinaryOp('*', other, self)

    def __truediv__(self, other):
        return PendingBinaryOp('/', self, other)

    def __rtruediv__(self, other):
        return PendingBinaryOp('/', other, self)

    def __pow__(self, other):
        return PendingBinaryOp('**', self, other)


class PendingFunctionCall:
    """A function call node in a pending expression tree."""

    def __init__(self, func_name, *args):
        self.func_name = func_name
        self.args = args

    def __repr__(self):
        return f"PendingFunctionCall({self.func_name!r}, {self.args!r})"

    def __neg__(self):
        return PendingUnaryOp('neg', self)

    def __add__(self, other):
        return PendingBinaryOp('+', self, other)

    def __radd__(self, other):
        return PendingBinaryOp('+', other, self)

    def __sub__(self, other):
        return PendingBinaryOp('-', self, other)

    def __rsub__(self, other):
        return PendingBinaryOp('-', other, self)

    def __mul__(self, other):
        return PendingBinaryOp('*', self, other)

    def __rmul__(self, other):
        return PendingBinaryOp('*', other, self)

    def __truediv__(self, other):
        return PendingBinaryOp('/', self, other)

    def __rtruediv__(self, other):
        return PendingBinaryOp('/', other, self)

    def __pow__(self, other):
        return PendingBinaryOp('**', self, other)


# Convenience functions mirroring math/numpy for use in ODE expressions

def exp(x):
    return PendingFunctionCall('exp', x)

def log(x):
    return PendingFunctionCall('log', x)

def log10(x):
    return PendingFunctionCall('log10', x)

def sqrt(x):
    return PendingFunctionCall('sqrt', x)

def sin(x):
    return PendingFunctionCall('sin', x)

def cos(x):
    return PendingFunctionCall('cos', x)

def tan(x):
    return PendingFunctionCall('tan', x)

def abs_(x):
    return PendingFunctionCall('abs', x)


_PENDING_TYPES = (KipetProxy, PendingBinaryOp, PendingUnaryOp, PendingFunctionCall)


def is_pending(expr):
    """Return True if expr is any kind of pending expression node."""
    return isinstance(expr, _PENDING_TYPES)


def resolve_expression(expr, subst: dict, functions=None):
    """
    Walk a pending expression tree and replace KipetProxy nodes
    with real values/discopt expressions from subst.

    Parameters
    ----------
    expr : pending tree node, int, or float
        The expression to resolve.
    subst : dict
        Mapping from variable name to resolved value (discopt Variable element or float).
    functions : dict, optional
        Override table for function-call nodes (``exp``, ``log``, ...). Pass
        JAX's ``jax.numpy`` functions to evaluate the tree for sequential
        integration (variable projection), where the substituted values are
        traced JAX arrays rather than discopt Variables. When given it is used
        for every function call, bypassing the discopt/scalar dispatch.

    Returns
    -------
    Resolved expression (discopt expression, JAX array, or float).
    """
    import discopt.modeling as dm

    BINARY_OPS = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b,
        '**': lambda a, b: a ** b,
    }

    import math

    DISCOPT_FUNCTIONS = {
        'exp': dm.exp,
        'log': dm.log,
        'log10': dm.log10,
        'sqrt': dm.sqrt,
        'sin': dm.sin,
        'cos': dm.cos,
        'tan': dm.tan,
        'abs': dm.abs,
    }

    # Fallback scalar functions for when args are plain floats/ints
    SCALAR_FUNCTIONS = {
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'abs': abs,
    }

    if isinstance(expr, KipetProxy):
        if expr.name not in subst:
            raise KeyError(f"Variable {expr.name!r} not found in substitution dict. "
                           f"Available: {list(subst.keys())}")
        return subst[expr.name]

    elif isinstance(expr, (int, float)):
        return expr

    elif isinstance(expr, PendingUnaryOp):
        operand = resolve_expression(expr.operand, subst, functions)
        if expr.op == 'neg':
            return -operand
        else:
            raise ValueError(f"Unknown unary op: {expr.op!r}")

    elif isinstance(expr, PendingBinaryOp):
        L = resolve_expression(expr.left, subst, functions)
        R = resolve_expression(expr.right, subst, functions)
        if expr.op not in BINARY_OPS:
            raise ValueError(f"Unknown binary op: {expr.op!r}")
        return BINARY_OPS[expr.op](L, R)

    elif isinstance(expr, PendingFunctionCall):
        args = [resolve_expression(a, subst, functions) for a in expr.args]
        func_name = expr.func_name
        if functions is not None:
            if func_name not in functions:
                raise ValueError(f"Unknown function: {func_name!r}")
            return functions[func_name](*args)
        if func_name not in DISCOPT_FUNCTIONS:
            raise ValueError(f"Unknown function: {func_name!r}")
        # If all args are plain numbers, use scalar math to avoid discopt Expression overhead
        if all(isinstance(a, (int, float)) for a in args):
            return SCALAR_FUNCTIONS[func_name](*args)
        return DISCOPT_FUNCTIONS[func_name](*args)

    else:
        # Could be a raw discopt expression or numeric - pass through
        return expr
