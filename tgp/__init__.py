"""The Graph Pooling library for PyTorch Geometric."""

import importlib
import sys

eps = 1e-8

__version__ = "0.3.0"

# List of submodules you want to allow lazy importing
_submodules = [
    "poolers",
    "src",
    "select",
    "reduce",
    "lift",
    "connect",
    "datasets",
    "data",
    "mp",
    "transforms",
    "utils",
]


def __getattr__(name):
    if name in _submodules:
        if name == "transforms":
            module = importlib.import_module(".data.transforms", __name__)
        else:
            module = importlib.import_module(f".{name}", __name__)
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
