"""Minimal Sphinx configuration for BenchMFG."""

project = "BenchMFG"
extensions = ["myst_nb"]  # myst_nb bundles myst_parser and renders .ipynb
nb_execution_mode = "off"  # notebooks are committed with their outputs
html_theme = "sphinx_rtd_theme"
html_title = "BenchMFG documentation"
html_logo = "../favicon_v3.svg"
exclude_patterns = ["_build", "**/gifs"]
