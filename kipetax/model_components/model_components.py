"""
Model elements: the named quantities a ReactionModel is built from.

Each element is a small record (name, value, bounds and a few kind-specific
options) plus a :class:`~kipetax.model_tools.proxy_vars.KipetProxy` that stands
in for it inside ODE and algebraic expressions.

The kinds differ only in which options they carry and what those default to,
so each subclass is described by a table (``_options``) rather than by its own
constructor. Options not in the table are accepted and kept in ``extra`` so
that keyword arguments meaningful elsewhere (``units=...``, for example) do not
raise.
"""
from kipetax.model_tools.proxy_vars import KipetProxy

_UNBOUNDED = (None, None)


def _validated_name(name):
    """Return ``name`` as a string, or raise if it cannot name a variable.

    Element names become proxy-variable names, results-table columns and
    dictionary keys, so they must be usable as Python identifiers.
    """
    name = str(name)
    if not name.isidentifier():
        raise ValueError(
            f"{name!r} is not a valid element name; use letters, digits and "
            f"underscores, not starting with a digit (e.g. 'A_1', not '1-A').")
    return name


class ModelElement:
    """Base class for all model elements.

    Subclasses set ``kind`` (the proxy-variable kind) and ``_options`` (a
    mapping of option name to default value).
    """

    kind = None
    _options = {}

    def __init__(self, name, value=None, description=None, **options):
        self.name = _validated_name(name)
        self.value = value
        self.description = description
        for option, default in self._options.items():
            setattr(self, option, options.pop(option, default))
        self.extra = options
        self.proxy_var = KipetProxy(self.name, self.kind)

    @property
    def lb(self):
        """Lower bound, or None if unbounded below."""
        return getattr(self, 'bounds', _UNBOUNDED)[0]

    @property
    def ub(self):
        """Upper bound, or None if unbounded above."""
        return getattr(self, 'bounds', _UNBOUNDED)[1]

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, value={self.value!r})"


class ModelParameter(ModelElement):
    """A kinetic parameter, estimated unless ``fixed``.

    With no ``value`` but finite bounds, the starting value is the midpoint of
    the bounds.
    """

    kind = 'parameter'
    _options = {'bounds': _UNBOUNDED, 'fixed': False, 'variance': None}

    def __init__(self, name, value=None, **options):
        super().__init__(name, value, **options)
        lo, hi = self.bounds
        if self.value is None and lo is not None and hi is not None:
            self.value = 0.5 * (lo + hi)


class ModelComponent(ModelElement):
    """A chemical species whose state is its concentration.

    ``value`` is the initial concentration. ``absorbing`` controls whether the
    species contributes to a Beer-Lambert spectral model.
    """

    kind = 'component'
    _options = {'bounds': _UNBOUNDED, 'variance': None, 'known': True,
                'absorbing': True, 'inert': False, 'S': None}

    def __init__(self, name, value=None, **options):
        options.pop('state', None)   # a component is always a concentration
        super().__init__(name, value, **options)
        self.state = 'concentration'


class ModelState(ModelElement):
    """A differential state that is not a concentration (volume, temperature)."""

    kind = 'state'
    _options = {'state': 'state', 'bounds': _UNBOUNDED, 'variance': None,
                'known': True, 'data': None}


class ModelAlgebraic(ModelElement):
    """An algebraic variable: no time derivative, optionally fixed to ``data``."""

    kind = 'algebraic'
    _options = {'bounds': _UNBOUNDED, 'data': None, 'step': None,
                'is_reaction': False, 'active': True, 'state': None}


class ModelConstant(ModelElement):
    """A fixed numeric constant used in expressions; never estimated."""

    kind = 'constant'
