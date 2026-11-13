"""
Location-tolerant imports for discopt internals.

kipetax reaches into two discopt private modules: the NLP evaluator and the
parametric-solve helper. discopt renamed the package that holds them from
``discopt._jax`` to ``discopt._relax``, which broke every collocation solve in
kipetax at once -- the import sits on the hot path of ``solve_local_nlp``, so
the failure is total rather than degraded.

Resolving the symbols by search instead of by a hard-coded path means a rename
of a private module (which discopt is free to do without notice, precisely
because it is private) costs one entry in a tuple here rather than a broken
release. The candidates are ordered newest-first.

If discopt ever exposes these publicly, drop this module and import directly.
"""

_EVALUATOR_MODULES = (
    'discopt._relax.nlp_evaluator',
    'discopt._jax.nlp_evaluator',
)

_PARAMETRIC_MODULES = (
    'discopt._relax.parametric',
    'discopt.parametric',
    'discopt._jax.parametric',
)


def _resolve(candidates, attr):
    """First ``attr`` found across ``candidates``, else a listing of the misses."""
    import importlib

    tried = []
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            tried.append(f'{mod_name} (no such module)')
            continue
        obj = getattr(mod, attr, None)
        if obj is not None:
            return obj
        tried.append(f'{mod_name} (no attribute {attr!r})')

    raise ImportError(
        f"could not locate {attr!r} in the installed discopt. Tried: "
        + '; '.join(tried)
        + ". discopt moves its private modules; add the new location to "
          "kipetax/model_tools/discopt_compat.py."
    )


def get_nlp_evaluator():
    """The ``NLPEvaluator`` class from whichever discopt layout is installed."""
    return _resolve(_EVALUATOR_MODULES, 'NLPEvaluator')


def get_extract_x_flat():
    """The ``extract_x_flat`` helper from whichever discopt layout is installed."""
    return _resolve(_PARAMETRIC_MODULES, 'extract_x_flat')
