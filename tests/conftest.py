"""Pytest configuration and fixtures for torch-geometric-pool tests."""

import pytest

# Import the module to check availability
try:
    import torch_sparse
    from torch_sparse import SparseTensor

    HAS_TORCH_SPARSE = True
except ImportError:
    torch_sparse = None
    SparseTensor = None
    HAS_TORCH_SPARSE = False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "torch_sparse: mark test as requiring torch_sparse (deselect with '-m \"not torch_sparse\"')",
    )
    config.addinivalue_line(
        "markers",
        "no_torch_sparse: mark test as requiring torch_sparse to be NOT installed",
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests that import torch_sparse at module level."""
    for item in items:
        # Check if the test file imports torch_sparse at module level
        try:
            test_module = item.module
            if hasattr(test_module, "__file__"):
                with open(test_module.__file__, "r") as f:
                    content = f.read()
                    # Check for direct import of torch_sparse or SparseTensor at module level
                    # (not inside functions)
                    lines = content.split("\n")
                    for i, line in enumerate(lines[:50]):  # Check first 50 lines
                        stripped = line.strip()
                        # Skip comments and docstrings
                        if (
                            stripped.startswith("#")
                            or stripped.startswith('"""')
                            or stripped.startswith("'''")
                        ):
                            continue
                        # Check for imports
                        if "from torch_sparse import" in line or (
                            "import torch_sparse" in line
                            and not line.strip().startswith("#")
                        ):
                            # Only mark if not already marked
                            if not any(
                                mark.name == "torch_sparse"
                                for mark in item.iter_markers()
                            ):
                                item.add_marker(pytest.mark.torch_sparse)
                            break
        except (AttributeError, FileNotFoundError, IOError):
            pass


@pytest.fixture(scope="session")
def has_torch_sparse():
    """Fixture to check if torch_sparse is available."""
    return HAS_TORCH_SPARSE


@pytest.fixture(scope="session")
def sparse_tensor_class():
    """Fixture to get SparseTensor class if available, None otherwise."""
    return SparseTensor


@pytest.fixture(scope="session")
def torch_sparse_module():
    """Fixture to get torch_sparse module if available, None otherwise."""
    return torch_sparse


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Skip tests marked with torch_sparse if torch_sparse is not available."""
    if any(mark.name == "torch_sparse" for mark in item.iter_markers()):
        if not HAS_TORCH_SPARSE:
            pytest.skip(
                "torch_sparse is not installed. Install with: pip install torch-sparse"
            )

    # Skip tests marked with no_torch_sparse if torch_sparse IS available
    if any(mark.name == "no_torch_sparse" for mark in item.iter_markers()):
        if HAS_TORCH_SPARSE:
            pytest.skip(
                "torch_sparse is installed. This test requires it to be uninstalled."
            )


# Helper function for tests to conditionally import SparseTensor
def get_sparse_tensor():
    """Get SparseTensor class if available, otherwise raise ImportError."""
    if not HAS_TORCH_SPARSE:
        pytest.skip("torch_sparse is not installed")
    return SparseTensor
