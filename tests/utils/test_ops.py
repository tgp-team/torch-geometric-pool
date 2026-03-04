"""Tests for tgp.utils.ops module."""

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import (
    erdos_renyi_graph,
    is_undirected,
    to_dense_adj,
    to_undirected,
)

from tests.test_utils import set_random_seed as _set_random_seed
from tgp.utils.ops import (
    add_remaining_self_loops,
    apply_dense_node_mask,
    batched_negative_edge_sampling,
    connectivity_to_sparsetensor,
    connectivity_to_torch_coo,
    create_one_hot_tensor,
    delta_gcn_matrix,
    dense_to_block_diag,
    edge_index_to_vector_id,
    expand_compacted_rows,
    get_assignments,
    get_mask_from_dense_s,
    negative_edge_sampling,
    postprocess_adj_pool_sparse,
    propagate_assignments_sparse,
    sample_almost_k_edges,
    vector_id_to_edge_index,
)


@pytest.mark.torch_sparse
def test_connectivity_to_torch_coo_with_sparsetensor_none_value():
    """Test connectivity_to_torch_coo with SparseTensor that has None edge values."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3

    sparse_adj = SparseTensor.from_edge_index(
        edge_index, edge_attr=None, sparse_sizes=(num_nodes, num_nodes)
    )
    row, _col, value = sparse_adj.coo()
    assert value is None

    result = connectivity_to_torch_coo(
        sparse_adj, edge_weight=None, num_nodes=num_nodes
    )

    assert isinstance(result, torch.Tensor)
    assert result.is_sparse
    assert result.shape == (num_nodes, num_nodes)

    result_values = result.values()
    assert result_values is not None
    assert result_values.size(0) == row.size(0)
    assert torch.all(result_values == 1.0)


@pytest.mark.torch_sparse
def test_connectivity_to_sparsetensor_with_sparsetensor_input():
    """Test connectivity_to_sparsetensor with SparseTensor as input."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    num_nodes = 3

    sparse_adj = SparseTensor.from_edge_index(
        edge_index, edge_attr=edge_weight, sparse_sizes=(num_nodes, num_nodes)
    )
    result = connectivity_to_sparsetensor(
        sparse_adj, edge_weight=None, num_nodes=num_nodes
    )

    assert isinstance(result, SparseTensor)
    assert result is sparse_adj


def test_connectivity_to_sparsetensor_import_error():
    """Test connectivity_to_sparsetensor raises ImportError when torch_sparse is not available."""
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

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3

    sparse_adj = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    )

    result_adj, result_weight = add_remaining_self_loops(
        sparse_adj, edge_weight=None, fill_value=1.0, num_nodes=num_nodes
    )

    assert isinstance(result_adj, SparseTensor)
    assert result_weight is None
    assert result_adj.size(0) == num_nodes
    assert result_adj.size(1) == num_nodes

    row, col, value = result_adj.coo()
    diagonal_mask = row == col
    assert diagonal_mask.sum() == num_nodes
    if value is not None:
        assert torch.allclose(value[diagonal_mask], torch.tensor(1.0))


@pytest.mark.torch_sparse
def test_add_remaining_self_loops_with_sparsetensor_resize():
    """Test add_remaining_self_loops with SparseTensor input and num_nodes requiring resize."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    original_num_nodes = 3
    new_num_nodes = 5

    sparse_adj = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(original_num_nodes, original_num_nodes)
    )

    result_adj, result_weight = add_remaining_self_loops(
        sparse_adj, edge_weight=None, fill_value=1.0, num_nodes=new_num_nodes
    )

    assert isinstance(result_adj, SparseTensor)
    assert result_weight is None
    assert result_adj.size(0) == new_num_nodes
    assert result_adj.size(1) == new_num_nodes


def test_add_remaining_self_loops_with_torch_coo():
    """Test add_remaining_self_loops with torch sparse COO tensor input."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    num_nodes = 3

    sparse_adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()
    result_adj, result_weight = add_remaining_self_loops(
        sparse_adj, edge_weight=None, fill_value=1.0, num_nodes=num_nodes
    )

    assert isinstance(result_adj, torch.Tensor)
    assert result_adj.is_sparse
    assert result_weight is None
    assert result_adj.shape == (num_nodes, num_nodes)

    result_indices = result_adj.indices()
    result_values = result_adj.values()
    for i in range(num_nodes):
        mask = (result_indices[0] == i) & (result_indices[1] == i)
        assert mask.any(), f"Node {i} should have a self-loop"
        self_loop_value = result_values[mask]
        assert torch.allclose(self_loop_value, torch.tensor(1.0))


def test_delta_gcn_matrix_with_torch_coo():
    """Test delta_gcn_matrix with torch sparse COO tensor input."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    num_nodes = 3

    sparse_adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()
    result_adj, result_weight = delta_gcn_matrix(
        sparse_adj, edge_weight=None, delta=2.0, num_nodes=num_nodes
    )

    assert isinstance(result_adj, torch.Tensor)
    assert result_adj.is_sparse
    assert result_weight is None
    assert result_adj.shape == (num_nodes, num_nodes)
    assert result_adj._nnz() > 0


def test_dense_to_block_diag_2d_and_invalid_dim():
    adj_2d = torch.tensor([[1.0, 0.0], [0.5, 2.0]])
    edge_index, edge_weight = dense_to_block_diag(adj_2d)
    assert edge_index.shape[0] == 2
    assert edge_weight.numel() == edge_index.shape[1]

    with pytest.raises(ValueError, match="adj_pool must have shape"):
        dense_to_block_diag(torch.tensor([1.0, 2.0, 3.0]))


def test_get_mask_from_dense_s_validation_and_empty_batch_slot():
    with pytest.raises(ValueError, match="s must have shape"):
        get_mask_from_dense_s(torch.ones(1, 1, 1, 1))

    s = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]], dtype=torch.float32)
    batch = torch.tensor([0, 2, 2], dtype=torch.long)
    mask = get_mask_from_dense_s(s, batch=batch)

    assert mask.shape == (3, 2)
    assert torch.equal(mask[1], torch.tensor([False, False]))


def test_apply_dense_node_mask_validation_errors():
    with pytest.raises(ValueError, match="expects x to be 3D"):
        apply_dense_node_mask(torch.randn(2, 3), torch.ones(2, 3, dtype=torch.bool))

    with pytest.raises(ValueError, match="expects mask shape"):
        apply_dense_node_mask(
            torch.randn(2, 3, 4),
            torch.ones(2, 2, dtype=torch.bool),
        )


def test_expand_compacted_rows_validation_errors():
    with pytest.raises(ValueError, match="at least 1D"):
        expand_compacted_rows(torch.tensor(1.0), torch.tensor([True]), expected_rows=1)

    with pytest.raises(ValueError, match="must contain exactly 2 entries"):
        expand_compacted_rows(torch.randn(1, 3), None, expected_rows=2)

    with pytest.raises(ValueError, match="must contain exactly 2 entries"):
        expand_compacted_rows(
            torch.randn(1, 3),
            torch.tensor([True], dtype=torch.bool),
            expected_rows=2,
        )

    with pytest.raises(ValueError, match="x_compact has 2 rows"):
        expand_compacted_rows(
            torch.randn(2, 3),
            torch.tensor([True, False, False], dtype=torch.bool),
            expected_rows=3,
        )


def test_postprocess_adj_pool_sparse_filters_tiny_weights():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.0, 1.0], dtype=torch.float32)

    out_index, out_weight = postprocess_adj_pool_sparse(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=2,
        remove_self_loops=False,
        degree_norm=False,
        edge_weight_norm=False,
    )

    assert out_index.shape == (2, 1)
    assert out_weight.shape == (1,)
    assert torch.allclose(out_weight, torch.tensor([1.0]))


def test_connectivity_to_torch_coo_rejects_invalid_type():
    with pytest.raises(
        ValueError, match="Edge index must be of type Tensor or SparseTensor"
    ):
        connectivity_to_torch_coo("invalid-edge-index", num_nodes=2)


def test_connectivity_to_torch_coo_defensive_else_branch(monkeypatch):
    import tgp.utils.ops as ops

    class Dummy:
        pass

    calls = iter([True, False])
    monkeypatch.setattr(ops, "is_sparsetensor", lambda _x: next(calls))

    with pytest.raises(
        ValueError, match="Edge index must be a Tensor or SparseTensor."
    ):
        ops.connectivity_to_torch_coo(Dummy(), num_nodes=2)


def test_create_one_hot_tensor_scalar_and_explicit_dtype():
    scalar_kept = create_one_hot_tensor(
        num_nodes=4,
        kept_node_tensor=torch.tensor(2),
        device=torch.device("cpu"),
    )
    assert scalar_kept.shape == (4, 2)
    assert scalar_kept[2, 1] == 1

    vector_kept = create_one_hot_tensor(
        num_nodes=5,
        kept_node_tensor=torch.tensor([1, 3], dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert vector_kept.shape == (5, 3)
    assert vector_kept.dtype == torch.float64


def test_propagate_assignments_sparse_all_zero_best_assignments_branch():
    assignments = torch.tensor([2, 0], dtype=torch.long)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    kept_node_tensor = torch.tensor([0], dtype=torch.long)
    mask = torch.tensor([True, False], dtype=torch.bool)

    out_assignments, mapping, out_mask = propagate_assignments_sparse(
        assignments=assignments,
        edge_index=edge_index,
        kept_node_tensor=kept_node_tensor,
        mask=mask,
        num_clusters=1,
    )

    assert torch.equal(out_assignments, assignments)
    assert mapping.shape == (2, 0)
    assert torch.equal(out_mask, mask)


def test_get_assignments_with_torch_sparse_coo_input():
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    values = torch.ones(2, dtype=torch.float32)
    edge_index_sparse = torch.sparse_coo_tensor(indices, values, size=(2, 2)).coalesce()

    assignments = get_assignments(
        kept_node_indices=[0],
        edge_index=edge_index_sparse,
        max_iter=1,
        num_nodes=2,
    )

    assert assignments.shape == (2, 2)
    assert assignments[0].tolist() == [0, 1]


@pytest.fixture
def set_random_seed():
    _set_random_seed(42)


def is_negative(edge_index, neg_edge_index, size, bipartite):
    adj = torch.zeros(size, dtype=torch.bool)
    neg_adj = torch.zeros(size, dtype=torch.bool)

    adj[edge_index[0], edge_index[1]] = True
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    if not bipartite:
        arange = torch.arange(size[0])
        assert neg_adj[arange, arange].sum() == 0

    return (adj & neg_adj).sum() == 0


def test_edge_index_to_vector_and_vice_versa(set_random_seed):
    n1, n2 = 13, 17
    row = torch.arange(n1).view(-1, 1).repeat(1, n2).view(-1)
    col = torch.arange(n2).view(1, -1).repeat(n1, 1).view(-1)
    edge_index = torch.stack([row, col], dim=0)

    idx = edge_index_to_vector_id(edge_index, (n1, n2))
    assert idx.tolist() == list(range(n1 * n2))
    edge_index2 = torch.stack(vector_id_to_edge_index(idx, (n1, n2)), dim=0)
    assert edge_index.tolist() == edge_index2.tolist()

    vector_id = torch.arange(n1 * n2)
    edge_index3 = torch.stack(vector_id_to_edge_index(vector_id, (n1, n2)), dim=0)
    assert edge_index.tolist() == edge_index3.tolist()


def test_dense_negative_edge_sampling(set_random_seed):
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_edge_sampling(edge_index, method="dense")
    assert neg_edge_index.size(1) == edge_index.size(1)
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    neg_edge_index = negative_edge_sampling(
        edge_index, method="dense", num_neg_samples=2
    )
    assert neg_edge_index.size(1) == 2
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    edge_index = to_undirected(edge_index)
    neg_edge_index = negative_edge_sampling(
        edge_index, method="dense", force_undirected=True
    )
    assert neg_edge_index.size(1) == edge_index.size(1) - 1
    assert is_undirected(neg_edge_index)
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)


def test_dense_bipartite_negative_edge_sampling(set_random_seed):
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_edge_sampling(
        edge_index, method="dense", num_nodes=(3, 4)
    )
    assert neg_edge_index.size(1) == edge_index.size(1)
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)

    neg_edge_index = negative_edge_sampling(
        edge_index, num_nodes=(3, 4), num_neg_samples=2, method="dense"
    )
    assert neg_edge_index.size(1) == 2
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)


def test_negative_edge_sampling_with_different_edge_density(set_random_seed):
    for num_nodes in [10, 100, 1000]:
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for is_directed in [False, True]:
                edge_index = erdos_renyi_graph(num_nodes, p, is_directed)
                neg_edge_index = negative_edge_sampling(
                    edge_index, num_nodes, force_undirected=not is_directed
                )
                assert is_negative(
                    edge_index, neg_edge_index, (num_nodes, num_nodes), bipartite=False
                )


def test_sparse_negative_edge_sampling_warns_on_dense_graph(set_random_seed):
    num_nodes = 4
    total_edges = num_nodes * num_nodes
    edge_ids = torch.arange(total_edges - 4)
    row, col = vector_id_to_edge_index(edge_ids, (num_nodes, num_nodes))
    edge_index = torch.stack([row, col], dim=0)

    with pytest.warns(
        UserWarning,
        match="The probability of sampling a negative edge is too low",
    ):
        _ = negative_edge_sampling(
            edge_index, num_nodes=num_nodes, num_neg_samples=2, method="sparse"
        )


def test_bipartite_negative_edge_sampling_with_different_edge_density():
    for num_nodes in [10, 100, 1000]:
        for p in [0.1, 0.3, 0.5, 0.8]:
            size = (num_nodes, int(num_nodes * 1.2))
            n_edges = int(p * size[0] * size[1])
            row, col = (
                torch.randint(size[0], (n_edges,)),
                torch.randint(size[1], (n_edges,)),
            )
            edge_index = torch.stack([row, col], dim=0)
            neg_edge_index = negative_edge_sampling(edge_index, size)
            assert is_negative(edge_index, neg_edge_index, size, bipartite=True)


def test_dense_batched_negative_edge_sampling(set_random_seed):
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    neg_edge_index = batched_negative_edge_sampling(edge_index, batch, method="dense")
    assert neg_edge_index.size(1) == edge_index.size(1)

    adj = torch.zeros(8, 8, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    neg_adj = torch.zeros(8, 8, dtype=torch.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    assert (adj & neg_adj).sum() == 0
    assert (adj | neg_adj).sum() == edge_index.size(1) + neg_edge_index.size(1)
    assert neg_adj[:4, 4:].sum() == 0
    assert neg_adj[4:, :4].sum() == 0


def test_sparse_batched_negative_edge_sampling(set_random_seed):
    num_nodes_per_graph = [100, 75, 220]
    graph_list = [
        Data(edge_index=erdos_renyi_graph(n, edge_prob=0.2))
        for n in num_nodes_per_graph
    ]
    batched_data = Batch.from_data_list(graph_list)
    batch_size = batched_data.batch_size
    edge_index, batch = batched_data.edge_index, batched_data.batch

    neg_edge_index = batched_negative_edge_sampling(edge_index, batch)
    assert neg_edge_index.size(1) <= edge_index.size(1)

    adj = to_dense_adj(
        edge_index, batch, batch_size=batch_size, max_num_nodes=max(num_nodes_per_graph)
    ).bool()
    neg_adj = to_dense_adj(
        neg_edge_index,
        batch,
        batch_size=batch_size,
        max_num_nodes=max(num_nodes_per_graph),
    ).bool()

    assert (adj & neg_adj).sum() == 0
    assert (adj | neg_adj).sum() == edge_index.size(1) + neg_edge_index.size(1)


def test_bipartite_batched_negative_edge_sampling(set_random_seed):
    edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
    edge_index2 = edge_index1 + torch.tensor([[2], [4]])
    edge_index3 = edge_index2 + torch.tensor([[2], [4]])
    edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
    src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
    dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    neg_edge_index = batched_negative_edge_sampling(edge_index, (src_batch, dst_batch))
    assert neg_edge_index.size(1) <= edge_index.size(1)

    adj = torch.zeros(6, 12, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    neg_adj = torch.zeros(6, 12, dtype=torch.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    assert (adj & neg_adj).sum() == 0
    assert (adj | neg_adj).sum() == edge_index.size(1) + neg_edge_index.size(1)


def test_sample_almost_k_edges_caps_k():
    size = (2, 2)
    new_edge_index, new_edge_id = sample_almost_k_edges(
        size=size,
        k=10,
        force_undirected=False,
        remove_self_loops=False,
        method="sparse",
    )
    assert new_edge_id.numel() <= size[0] * size[1]
    assert new_edge_index.shape[0] == 2
