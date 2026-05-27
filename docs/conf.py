project = "MoE Competition Documentation"
author = "Huawei Zurich CPL"
copyright = "2026, Huawei Zurich CPL"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosectionlabel_prefix_document = True
myst_heading_anchors = 3
