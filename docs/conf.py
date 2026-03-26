import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'Agent Actuariel'
author = 'Actuariel'
release = '1.0'
language = 'fr'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.mermaid',
]

html_theme = 'sphinx_rtd_theme'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
