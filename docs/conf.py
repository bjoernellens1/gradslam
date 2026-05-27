# Configuration file for the Sphinx documentation builder.
#
# Modernized for gradslam ROCm/PyTorch rewrite (2026)

import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    gradslam_version = metadata.version("opengradslam")
except metadata.PackageNotFoundError:
    gradslam_version = "0.0.0+editable"

# The master toctree document.
master_doc = 'index'

# -- Project information

project = 'gradslam'
copyright = '2020–2026, Montreal Robotics Lab'
author = 'MontrealRobotics'
version = gradslam_version
release = gradslam_version

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
]

autosectionlabel_prefix_document = True

# Napoleon (Google-style docstrings)
napoleon_use_ivar = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc options
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autosummary_generate = True

# Mock imports for dependencies not on RTD
autodoc_mock_imports = [
    'open3d',
    'trimesh',
    'scikit_image',
]

# nbsphinx (Jupyter notebook rendering)
if os.environ.get('READTHEDOCS') == 'True':
    nbsphinx_execute = 'never'
else:
    nbsphinx_execute = 'auto'
    nbsphinx_timeout = 60

# Paths
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Syntax highlighting
pygments_style = 'sphinx'

# Options
add_module_names = False
todo_include_todos = False

# -- HTML output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
}

_STATIC_IMG_DIR = os.path.join(os.path.dirname(__file__), "_static", "img")

if os.path.exists(os.path.join(_STATIC_IMG_DIR, "gradslam-logo.png")):
    html_logo = '_static/img/gradslam-logo.png'
if os.path.exists(os.path.join(_STATIC_IMG_DIR, "gradslam-favicon-32x32.png")):
    html_favicon = '_static/img/gradslam-favicon-32x32.png'

html_static_path = ['_static']

# -- LaTeX output

latex_elements = {
    'fontpkg': r'\usepackage{amsmath, amsfonts, amssymb, amsthm}',
}

latex_documents = [
    (master_doc, 'gradslam.tex', 'gradslam Documentation', author, 'manual'),
]

# -- Intersphinx

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'kornia': ('https://kornia.readthedocs.io/en/latest', None),
}
