"""
kipetax - Kinetic Parameter Estimation Toolkit (Pyomo-free rewrite using discopt).

Primary exports:
  ReactionModel  : Define and solve kinetic models
  ReactionSet    : Manage multiple experiments with shared parameters
"""
# Enable JAX float64 precision required for accurate ODE integration.
# Must be set before JAX is first imported.
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')

from kipetax.main_modules.reaction_model import ReactionModel
from kipetax.main_modules.reaction_set import ReactionSet
from kipetax.main_modules.population_balance import PopulationBalanceModel
from kipetax.main_modules.mixture_model import MixtureModel

# Proxy math functions available in ODE expressions
from kipetax.model_tools.proxy_vars import exp, log, sqrt, sin, cos, tan, log10, abs_

__version__ = '0.1.0'
__all__ = [
    'ReactionModel',
    'ReactionSet',
    'PopulationBalanceModel',
    'MixtureModel',
    'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'log10', 'abs_',
]
