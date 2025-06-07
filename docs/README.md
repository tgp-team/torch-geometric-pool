# Building Documentation

To build the documentation from <img src="source/_static/img/tgp-logo.svg" width="20px" align="center" style="display: inline-block; height: 1.0em; width: unset; vertical-align: text-top;"/> tgp root directory:

1. Install PyTorch and PyG via `pip install -r docs/requirements.txt`.
2. Install `tgp` and [Sphinx](https://www.sphinx-doc.org/en/master/) requirements
   via `pip install ".[doc]"`
3. Generate the documentation file via:

```bash
cd docs
make html
```

The documentation is now available to view by opening
`docs/build/html/index.html`.
