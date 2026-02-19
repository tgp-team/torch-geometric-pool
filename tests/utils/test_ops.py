"""Tests for tgp.utils.ops module."""

from unittest.mock import patch

import pytest
import torch

from tgp.utils.ops import (
    add_remaining_self_loops,
    connectivity_to_sparsetensor,
    connectivity_to_torch_coo,
    delta_gcn_matrix,
)


@pytest.mark.torch_sparse
def test_connectivity_to_torch_coo_with_sparsetensor_none_value():
    """Test connectivity_to_torch_coo with SparseTensor that has None edge values."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    # Create edge_index
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3

    # Create SparseTensor without edge_attr (value will be None)
    sparse_adj = SparseTensor.from_edge_index(
        edge_index, edge_attr=None, sparse_sizes=(num_nodes, num_nodes)
    )

    # Verify that coo() returns None for value
    row, col, value = sparse_adj.coo()
    assert value is None

    # Call connectivity_to_torch_coo
    result = connectivity_to_torch_coo(
        sparse_adj, edge_weight=None, num_nodes=num_nodes
    )

    # Verify result is a torch COO sparse tensor
    assert isinstance(result, torch.Tensor)
    assert result.is_sparse

    # Verify shape
    assert result.shape == (num_nodes, num_nodes)

    # Verify that values were set to ones (since value was None)
    result_values = result.values()
    assert result_values is not None
    assert result_values.size(0) == row.size(0)
    assert torch.all(result_values == 1.0)


@pytest.mark.torch_sparse
def test_connectivity_to_sparsetensor_with_sparsetensor_input():
    """Test connectivity_to_sparsetensor with SparseTensor as input."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    # Create edge_index
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    num_nodes = 3

    # Create SparseTensor
    sparse_adj = SparseTensor.from_edge_index(
        edge_index, edge_attr=edge_weight, sparse_sizes=(num_nodes, num_nodes)
    )

    # Call connectivity_to_sparsetensor with SparseTensor input
    result = connectivity_to_sparsetensor(
        sparse_adj, edge_weight=None, num_nodes=num_nodes
    )

    # Verify result is the same SparseTensor (should be returned as-is)
    assert isinstance(result, SparseTensor)
    assert result is sparse_adj  # Should be the same object


def test_connectivity_to_sparsetensor_import_error():
    """Test connectivity_to_sparsetensor raises ImportError when torch_sparse is not available."""
    # Mock HAS_TORCH_SPARSE to be False
    with patch("tgp.utils.ops.HAS_TORCH_SPARSE", False):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(
            ImportError,
            match="Cannot convert connectivity to sparse tensor: torch_sparse is not installed",
        ):
            connectivity_to_sparsetensor(edge_index)


@pytest.mark.torch_sparse
def test_add_remaining_self_loops_with_sparsetensor():
    """Test add_remaining_self_loops with SparseTensor input."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    # Create edge_index
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3

    # Create SparseTensor
    sparse_adj = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    )

    # Call add_remaining_self_loops with SparseTensor
    result_adj, result_weight = add_remaining_self_loops(
        sparse_adj, edge_weight=None, fill_value=1.0, num_nodes=num_nodes
    )

    # Verify result is a SparseTensor
    assert isinstance(result_adj, SparseTensor)
    assert result_weight is None

    # Verify shape
    assert result_adj.size(0) == num_nodes
    assert result_adj.size(1) == num_nodes

    # Verify that self-loops were added by checking the diagonal
    # Convert to COO format and check for diagonal elements
    row, col, value = result_adj.coo()
    diagonal_mask = row == col
    assert diagonal_mask.sum() == num_nodes  # All nodes should have self-loops
    # Verify the diagonal values are the fill_value (1.0)
    if value is not None:
        assert torch.allclose(value[diagonal_mask], torch.tensor(1.0))


@pytest.mark.torch_sparse
def test_add_remaining_self_loops_with_sparsetensor_resize():
    """Test add_remaining_self_loops with SparseTensor input and num_nodes requiring resize."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    # Create edge_index for a 3-node graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    original_num_nodes = 3
    new_num_nodes = 5  # Different from original

    # Create SparseTensor with original size
    sparse_adj = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(original_num_nodes, original_num_nodes)
    )

    # Call add_remaining_self_loops with different num_nodes (should trigger resize)
    result_adj, result_weight = add_remaining_self_loops(
        sparse_adj, edge_weight=None, fill_value=1.0, num_nodes=new_num_nodes
    )

    # Verify result is a SparseTensor
    assert isinstance(result_adj, SparseTensor)
    assert result_weight is None

    # Verify shape was resized
    assert result_adj.size(0) == new_num_nodes
    assert result_adj.size(1) == new_num_nodes


def test_add_remaining_self_loops_with_torch_coo():
    """Test add_remaining_self_loops with torch sparse COO tensor input."""
    # Create edge_index
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    num_nodes = 3

    # Convert to torch sparse COO tensor
    sparse_adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()

    # Call add_remaining_self_loops with torch COO sparse tensor
    result_adj, result_weight = add_remaining_self_loops(
        sparse_adj, edge_weight=None, fill_value=1.0, num_nodes=num_nodes
    )

    # Verify result is a torch sparse COO tensor
    assert isinstance(result_adj, torch.Tensor)
    assert result_adj.is_sparse
    assert result_weight is None

    # Verify shape
    assert result_adj.shape == (num_nodes, num_nodes)

    # Verify that self-loops were added
    # Get the indices and values
    result_indices = result_adj.indices()
    result_values = result_adj.values()

    # Check that all nodes have self-loops (diagonal elements)
    for i in range(num_nodes):
        # Find self-loop for node i
        mask = (result_indices[0] == i) & (result_indices[1] == i)
        assert mask.any(), f"Node {i} should have a self-loop"
        # Verify the self-loop has the fill_value
        self_loop_value = result_values[mask]
        assert torch.allclose(self_loop_value, torch.tensor(1.0))


def test_delta_gcn_matrix_with_torch_coo():
    """Test delta_gcn_matrix with torch sparse COO tensor input."""
    # Create edge_index
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    num_nodes = 3

    # Convert to torch sparse COO tensor
    sparse_adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()

    # Call delta_gcn_matrix with torch COO sparse tensor
    result_adj, result_weight = delta_gcn_matrix(
        sparse_adj, edge_weight=None, delta=2.0, num_nodes=num_nodes
    )

    # Verify result is a torch sparse COO tensor (same type as input)
    assert isinstance(result_adj, torch.Tensor)
    assert result_adj.is_sparse
    assert result_weight is None

    # Verify shape
    assert result_adj.shape == (num_nodes, num_nodes)

    # Verify it's a valid sparse tensor
    assert result_adj._nnz() > 0
