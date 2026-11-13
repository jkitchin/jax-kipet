"""
Helper utilities for kipetax (pure Python, no Pyomo).
"""
from collections import namedtuple


DosingPoint = namedtuple('DosingPoint', ['component', 'time', 'concentration', 'volume'])


class AttrDict(dict):
    """A dict that allows attribute-style access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"AttrDict has no attribute {name!r}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"AttrDict has no attribute {name!r}")
