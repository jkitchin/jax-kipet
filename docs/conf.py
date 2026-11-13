"""Sphinx configuration for kipetax documentation."""
import os

# Enable JAX float64 precision before kipetax (and JAX) are imported by autodoc.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# -- Project information -----------------------------------------------------
project = "kipetax"
author = "John Kitchin"
copyright = "2026, John Kitchin"

from importlib.metadata import version as _version

try:
    release = _version("kipetax")
except Exception:
    release = "2.0.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # NumPy / Google style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",               # Markdown support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Don't fail the build if optional/heavy deps are missing at doc-build time.
autodoc_mock_imports = ["discopt", "jax", "jaxlib"]

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"kipetax {release}"
html_static_path = ["_static"]
