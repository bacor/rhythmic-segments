# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Rhythmic Segments"
copyright = "2025, Bas Cornelissen"
author = "Bas Cornelissen"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",  # notebooks & Markdown parsing
    "sphinx_copybutton",
    # "sphinx_design",
    "autoapi.extension",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
html_theme_options = {}
html_css_files = ["custom.css"]


# MyST/Markdown
myst_enable_extensions = ["colon_fence", "deflist", "linkify", "dollarmath", "amsmath"]

# Notebooks: execute during build (set to "off" to trust existing outputs)
nb_execution_mode = "off"  # "off" | "auto"
nb_execution_timeout = 120

# AutoAPI: point to your package
autoapi_type = "python"
autoapi_dirs = ["../../src/rhythmic_segments"]  # adjust path
autoapi_add_toctree_entry = False
autoapi_keep_files = False

autoapi_root = "autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "bysource"
