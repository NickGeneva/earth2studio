# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for earth2bufr documentation."""

from __future__ import annotations

# -- Project information -----------------------------------------------------
project = "earth2bufr"
copyright = "2025 NVIDIA CORPORATION & AFFILIATES"  # noqa: A001
author = "NVIDIA Corporation"
version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "numpydoc",
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ["../src/earth2bufr"]
autoapi_type = "python"
autoapi_ignore = ["*/_types.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = True
autoapi_python_use_implicit_namespaces = False

# -- Napoleon / numpydoc -----------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
numpydoc_show_class_members = False

# -- HTML output -------------------------------------------------------------
html_theme = "nvidia_sphinx_theme"
html_title = "earth2bufr"

# -- Miscellaneous -----------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
