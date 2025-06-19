import datetime
import doctest
import os

from docutils import nodes

import tgp

os.environ["PYTORCH_JIT"] = '0'  # generate doc for torch.jit.script methods

# -- Project information -----------------------------------------------------
#

project = "Torch Geometric Pool"
author = "Filippo Maria Bianchi, Ivan Marisca"
copyright = "Copyright &copy; {}, {}".format(datetime.datetime.now().year, author)

version = tgp.__version__
release = version

# -- General configuration ---------------------------------------------------
#

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.doctest',
    "sphinx.ext.intersphinx",
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "sphinx_copybutton",
    "sphinx_design",
    'hoverxref.extension',
    'sphinx.ext.mathjax',
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "jupyter_sphinx",  # for running embedded code cells and showing their output
    'myst_nb',
    "sphinx_sitemap",
    # 'sphinxext.opengraph',
    # "sphinx_docsearch",
    # "sphinxcontrib.mermaid",
    # "sphinx_contributors",
]

autosummary_generate = True

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'

rst_context = {'tgp': tgp}

add_module_names = False
# autodoc_inherit_docstrings = False

napoleon_custom_sections = [("Shape", "params_style"),
                            ("Shapes", "params_style")]

numfig = True  # Enumerate figures and tables

# -- Options from Shibuya -------------------------------------------------
#

todo_include_todos = True
jupyter_sphinx_thebelab_config = {
    'requestKernel': True,
}
# jupyter_sphinx_require_url = ''  # uncomment if using sphinxcontrib.mermaid
sitemap_excludes = ['404/']

extlinks = {
    'pull': (
        'https://github.com/tgp-team/torch-geometric-pool/pull/%s',
        'pull request #%s'),
    'issue': (
        'https://github.com/tgp-team/torch-geometric-pool/issues/%s', 'issue #%s'),
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for intersphinx -------------------------------------------------
#

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ("https://numpy.org/devdocs/", None),
    'PyTorch': ('https://pytorch.org/docs/stable/', None),
    'PyG': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
    'PyG_v2.5': ('https://pytorch-geometric.readthedocs.io/en/2.5.2/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None)
}

# -- Hoverxref options -------------------------------------------------------
#

hoverxref_auto_ref = True
hoverxref_roles = ['class', 'mod', 'doc', 'meth', 'func']
hoverxref_mathjax = True
hoverxref_intersphinx = ['PyG', 'PyG_v2.5', 'PyTorch', 'numpy', 'scipy']

# -- myst_nb options -------------------------------------------------------
#

nb_execution_mode = 'auto' # off
myst_enable_extensions = ['dollarmath']
myst_dmath_allow_space = True
myst_dmath_double_inline = True
nb_execution_timeout = -1
# nb_code_prompt_hide = 'Hide code cell outputs'

# -- Theme options -----------------------------------------------------------
#

templates_path = ["_templates"]
html_static_path = ["_static"]
# html_extra_path = ["_public"]

html_title = "Torch Geometric Pool"
html_theme = "shibuya"
html_baseurl = ""
sitemap_url_scheme = "{link}"
language = "en"

html_copy_source = False
html_show_sourcelink = False

# html_additional_pages = {
#     "page": "page.html",
# }
html_css_files = [
    'css/custom.css',
]

if os.getenv('USE_DOCSEARCH'):
    extensions.append("sphinx_docsearch")
    docsearch_app_id = ""
    docsearch_api_key = ""
    docsearch_index_name = ""

if os.getenv("TRIM_HTML_SUFFIX"):
    html_link_suffix = ""

html_favicon = "_static/favicon.svg"

html_theme_options = {
    "logo_target": "/",
    "light_logo": "_static/img/tgp-logo-bar.svg",
    "dark_logo": "_static/img/tgp-logo-bar.svg",

    "accent_color": "violet",

    "og_image_url": "{url}/site_preview.jpg",
    "twitter_creator": "tgp",
    "twitter_site": "tgp",

    "discussion_url": "https://github.com/tgp-team/torch-geometric-pool/discussions",
    # "twitter_url": "https://twitter.com/tgp",
    # "discord_url": "https://discord.gg/tgp/",
    "github_url": "https://github.com/tgp-team/torch-geometric-pool",

    "carbon_ads_code": "",
    "carbon_ads_placement": "",

    "globaltoc_expand_depth": 0,
    "nav_links": [
        {
            "title": "What is pooling?",
            "url": "https://gnn-pooling.notion.site/",
            "external": True,
        },
    ]
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        # "sidebars/repo-stats.html",
        # "sidebars/edit-this-page.html",
        # "sidebars/carbon-ads.html",
        # "sidebars/ethical-ads.html",
    ]
}

if "READTHEDOCS" in os.environ:
    html_context = {
        "source_type": "github",
        "source_user": "tgp-team",
        "source_repo": "torch-geometric-pool",
    }
    html_theme_options["carbon_ads_code"] = ""
    html_theme_options["announcement"] = ""
else:
    html_context = {
        "source_type": "github",
        "source_user": "tgp-team",
        "source_repo": "torch-geometric-pool",
        "buysellads_code": "",
        "buysellads_placement": "",
        # "buysellads_container_selector": ".yue > section > section",
    }

DEBUG_RTD = False

if DEBUG_RTD:
    os.environ['READTHEDOCS_PROJECT'] = 'torch-geometric-pool'
    html_context["DEBUG_READTHEDOCS"] = True
    html_theme_options["carbon_ads_code"] = None


# -- Setup options -----------------------------------------------------------
#


def logo_role(name, rawtext, text, *args, **kwargs):
    if name == 'tgp':
        url = f'{html_baseurl}/_static/img/tgp-logo.svg'
    elif name in ['pyg', 'pytorch', 'lightning', 'hydra']:
        url = f'{html_baseurl}/_static/img/logos/{name}.svg'
    else:
        raise RuntimeError
    node = nodes.image(uri=url, alt=str(name).capitalize() + ' logo')
    classes = ['inline-logo', f'{name.lower().replace(" ", "-")}-inline-logo']
    node['classes'] += classes
    if text != 'null':
        node['classes'].append('with-text')
        span = nodes.inline(text=text)
        span['classes'] += [f'{node_cls}-text' for node_cls in classes]
        return [node, span], []
    return [node], []

def skip_member(app, what, name, obj, skip, options):
    # List of functions to skip in the doc
    if name in ("reset_parameters", "extra_repr_args", 
                "compute_loss", "reset_own_parameters"):
        return True
    return skip

def setup(app):
    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect("source-read", rst_jinja_render)
    app.connect("autodoc-skip-member", skip_member)

    app.add_role('tgp', logo_role)
    app.add_role('pyg', logo_role)
    app.add_role('pytorch', logo_role)
    app.add_role('hydra', logo_role)
    app.add_role('lightning', logo_role)
