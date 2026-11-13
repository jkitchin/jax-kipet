"""
pytest configuration for kipetax tests.

Sets JAX_ENABLE_X64=1 before any JAX imports to ensure float64 precision
is available for accurate ODE integration and NLP solving.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
