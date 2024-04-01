# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MAPLES-DR"
copyright = "2024, Gabriel Lepetit-Aimon"
author = "Gabriel Lepetit-Aimon"
release = "0.1alpha"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]


templates_path = ["_templates"]
exclude_patterns = []

# -- AutoDoc configuration ---------------------------------------------------
autodoc_typehints = "description"
autodoc_class_signature = "separated"
add_module_names = False


# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-napoleon
napoleon_numpy_docstring = True
# napoleon_preprocess_types = True
# napoleon_use_rtype = False

# -- intersphinx configuration ------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/icon.svg"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

# -- Sphinx Prolog -----------------------------------------------------------
rst_prolog = """
.. # define a hard line break for HTML
.. |br| raw:: html

   <br />
"""

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_prolog = """
.. raw:: html

    <style>
        .document p,
        .document h2,
        .document h3,
        .document h4,
        .document h5,
        .document h6 {
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .document .rendered_html pre{
            font-size: 12px;
        }
    </style>
"""

# -- Multi-language support --------------------------------------------------
# https://docs.readthedocs.io/en/stable/guides/manage-translations-sphinx.html
gettext_uuid = True
gettext_compact = False

for t in tags:
    if t.startswith("locales_"):
        language = t[8:]
        break
else:
    language = None

if language is not None:
    try:
        html_context
    except NameError:
        html_context = dict()
    html_context["display_lower_left"] = True

    # tell the theme which language to we're currently building
    html_context["current_language"] = language

    # POPULATE LINKS TO OTHER LANGUAGES
    html_context["languages"] = [("en", "../en/")]

    languages = [lang.name for lang in os.scandir("locales") if lang.is_dir()]
    for lang in languages:
        html_context["languages"].append((lang, "../" + lang + "/"))
else:
    language = "en"
