"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

# Add the source directory to the Python path
sys.path.insert(0, str(Path("../src").resolve()))

# Project information
project = "ScAN Gen"
copyright = "2025, ScAN Gen Contributors"
author = "ScAN Gen Contributors"
version = "0.2.0"
release = "0.2.0"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinxcontrib.mermaid",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# HTML theme
html_theme = "furo"
html_static_path = ["_static"]

# Auto-doc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
