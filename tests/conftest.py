"""Pytest configuration and fixtures for torch-geometric-pool tests."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Ensure the project root is in the Python path for imports
# This is needed for pytest to find the tests module
project_root = Path(__file__).parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Import test utilities using importlib for robust importing
# This works regardless of how pytest is invoked or the current working directory
test_utils_path = Path(__file__).parent / "test_utils.py"
if not test_utils_path.exists():
    raise ImportError(
        f"test_utils.py not found at {test_utils_path}. "
        f"Current file: {__file__}, Parent: {Path(__file__).parent}"
    )

spec = importlib.util.spec_from_file_location("test_utils", test_utils_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to create spec for test_utils.py at {test_utils_path}")

test_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_utils)

# Extract the functions we need
make_pooler_test_graph_dense = test_utils.make_pooler_test_graph_dense
make_pooler_test_graph_dense_batch = test_utils.make_pooler_test_graph_dense_batch
make_pooler_test_graph_sparse = test_utils.make_pooler_test_graph_sparse
make_pooler_test_graph_sparse_batch = test_utils.make_pooler_test_graph_sparse_batch

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


# Canonical pooler test graphs (same for all poolers)
@pytest.fixture(scope="session")
def pooler_test_graph_sparse():
    """Single sparse graph (x, edge_index, edge_weight, batch). Same for all poolers."""
    return make_pooler_test_graph_sparse(seed=42)


@pytest.fixture(scope="session")
def pooler_test_graph_dense():
    """Single dense graph (x, adj), B=1. Same for all poolers."""
    return make_pooler_test_graph_dense(seed=42)


@pytest.fixture(scope="session")
def pooler_test_graph_sparse_batch():
    """Batched sparse graph (Batch). Same for all poolers."""
    return make_pooler_test_graph_sparse_batch(seed=42)


@pytest.fixture(scope="session")
def pooler_test_graph_sparse_batch_tuple():
    """Batched sparse as (x, edge_index, edge_weight, batch). Same for all poolers."""
    import torch

    b = make_pooler_test_graph_sparse_batch(seed=42)
    ew = getattr(b, "edge_attr", None) or getattr(
        b, "edge_weight", torch.ones(b.edge_index.size(1))
    )
    return b.x, b.edge_index, ew, b.batch


@pytest.fixture(scope="session")
def pooler_test_graph_dense_batch():
    """Batched dense graph (x, adj). Same for all poolers."""
    return make_pooler_test_graph_dense_batch(seed=42)


@pytest.fixture(scope="session")
def pooler_test_graph_sparse_spt():
    """Single sparse graph as (x, adj_spt, edge_weight, batch); adj is SparseTensor. Same for all poolers."""
    if not HAS_TORCH_SPARSE:
        pytest.skip("torch_sparse required")
    x, edge_index, edge_weight, batch = make_pooler_test_graph_sparse(seed=42)
    adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)
    return x, adj, edge_weight, batch


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
