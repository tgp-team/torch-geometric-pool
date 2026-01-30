"""The Graph Pooling library for PyTorch Geometric."""

import importlib
import sys

eps = 1e-8

__version__ = "0.4.0"

# List of submodules you want to allow lazy importing
_submodules = [
    "poolers",
    "src",
    "select",
    "reduce",
    "lift",
    "connect",
    "datasets",
    "transforms",
    "utils",
]


def __getattr__(name):
    if name in _submodules:
        module = importlib.import_module(f".{name}", __name__)
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
