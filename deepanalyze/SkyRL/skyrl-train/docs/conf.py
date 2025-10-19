import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SkyRL"
copyright = "2025, NovaSkyAI"
author = "NovaSkyAI"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# External links configuration
extlinks = {
    "code_link": ("https://github.com/NovaSky-AI/skyrl/blob/main/skyrl-train/%s", "%s"),
    "skyrl_gym_link": (
        "https://github.com/NovaSky-AI/skyrl/blob/main/skyrl-gym/%s",
        "%s",
    ),
}
