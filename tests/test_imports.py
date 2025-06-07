import importlib
import sys

import pytest

import tgp
import tgp.imports as imp
from tgp import select


def test_check_torch_cluster_available_raises(monkeypatch):
    # Simulate torch_cluster missing
    monkeypatch.setattr(imp, "HAS_TORCH_CLUSTER", False)
    monkeypatch.setattr(imp, "torch_cluster", None)
    with pytest.raises(ImportError, match="The 'torch_cluster' package is required"):
        imp.check_torch_cluster_available()


def test_check_torch_sparse_available_raises(monkeypatch):
    # Simulate torch_sparse missing
    monkeypatch.setattr(imp, "HAS_TORCH_SPARSE", False)
    monkeypatch.setattr(imp, "torch_sparse", None)
    monkeypatch.setattr(imp, "SparseTensor", None)
    with pytest.raises(ImportError, match="The 'torch_sparse' package is required"):
        imp.check_torch_sparse_available()


def test_module_reload_resets_flags(monkeypatch):
    # Modify flags, reload, and verify attributes exist
    monkeypatch.setenv("IRRELEVANT", "VALUE")  # no-op
    importlib.reload(imp)
    # After reload, flags have been re-evaluated (but at least exist)
    assert hasattr(imp, "HAS_TORCH_CLUSTER")
    assert hasattr(imp, "HAS_TORCH_SPARSE")
    assert hasattr(imp, "HAS_TORCH_SCATTER")
    assert callable(imp.check_torch_cluster_available)
    assert callable(imp.check_torch_sparse_available)


def test_import_select():
    # Ensure the module has not already been imported
    if "tgp" in sys.modules:
        del sys.modules["tgp"]

    if "select" in sys.modules:
        del sys.modules["select"]

    # Ensure select module can be imported
    assert hasattr(select, "GraclusSelect")

    # Try to import a module that should not exist
    with pytest.raises(ImportError):
        from tgp import NonExistentModule  # noqa

    # Try to import a module that should not exist
    with pytest.raises(AttributeError):
        print(tgp.nonexistentmodule)


if __name__ == "__main__":
    pytest.main([__file__])
