import pytest
import torch

from tests.test_utils import make_chain_graph_sparse
from tgp.poolers import NMFPooling
from tgp.select.nmf_select import NMFSelect


def test_nmf_select(pooler_test_graph_dense):
    _, adj = pooler_test_graph_dense
    B, N, _ = adj.size()
    k = 2
    selector = NMFSelect(k=k)
    out = selector.forward(edge_index=adj.float())
    assert out.s.size(0) == B
    assert out.s.size(1) == N
    assert out.s.size(2) == k


def test_nmf_select_sparse_single_graph(pooler_test_graph_sparse):
    x, edge_index, edge_weight, _ = pooler_test_graph_sparse
    k = 3
    selector = NMFSelect(k=k)
    out = selector(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=x.size(0),
    )

    assert out.s.dim() == 2
    assert out.s.size(0) == x.size(0)
    assert out.s.size(1) <= k
    row_sums = out.s.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_nmf_select_sparse_single_graph_batch_infers_num_nodes():
    # Single graph with one isolated trailing node (node 3).
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    selector = NMFSelect(k=2)

    out = selector(edge_index=edge_index, batch=batch)

    assert out.s.dim() == 2
    assert out.s.size(0) == 4
    assert out.s.size(1) <= 2
    row_sums = out.s.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_nmf_select_sparse_single_graph_nonzero_batch_id():
    # Single graph encoded with a non-zero batch id.
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.full((4,), 5, dtype=torch.long)
    selector = NMFSelect(k=2)

    out = selector(edge_index=edge_index, batch=batch)

    assert out.s.dim() == 2
    assert out.s.size(0) == 4
    assert out.s.size(1) <= 2


def test_nmf_select_sparse_batched(pooler_test_graph_sparse_batch):
    data_batch = pooler_test_graph_sparse_batch
    k = 3
    selector = NMFSelect(k=k)
    out = selector(
        edge_index=data_batch.edge_index,
        edge_weight=data_batch.edge_attr,
        batch=data_batch.batch,
        num_nodes=data_batch.num_nodes,
    )

    assert out.s.dim() == 2
    assert out.s.size(0) == data_batch.num_nodes
    assert out.s.size(1) == k
    row_sums = out.s.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_nmf_select_k_larger_than_nodes_sparse():
    x, edge_index, edge_weight, _ = make_chain_graph_sparse(N=4, F_dim=3, seed=42)
    selector = NMFSelect(k=10)
    out = selector(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=x.size(0),
    )

    # Follow EigenPool-like behavior for single-graph sparse selection.
    assert out.s.size(1) <= x.size(0)


def test_nmf_unbatched_forward_dense_output(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    pooler = NMFPooling(k=3, batched=False, sparse_output=False)
    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)

    assert out.so is not None
    assert out.so.s.dim() == 2
    assert out.x.dim() == 3  # [B, K, F]
    assert out.edge_index.dim() == 3  # [B, K, K]
    assert out.edge_weight is None
    assert out.batch is not None
    assert out.batch.size(0) == out.so.num_supernodes


def test_nmf_unbatched_forward_sparse_output(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    pooler = NMFPooling(k=3, batched=False, sparse_output=True)
    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)

    assert out.so is not None
    assert out.so.s.dim() == 2
    assert out.x.dim() == 2  # [B*K, F]
    assert out.edge_index.dim() == 2 and out.edge_index.size(0) == 2
    assert out.edge_weight is not None
    assert out.batch is not None
    assert out.batch.size(0) == out.so.num_supernodes


def test_nmf_precoarsening(pooler_test_graph_sparse_batch):
    data_batch = pooler_test_graph_sparse_batch
    num_graphs = data_batch.num_graphs
    num_nodes = data_batch.num_nodes

    k = 3
    pooling_out = NMFPooling(k=k).precoarsening(
        edge_index=data_batch.edge_index,
        edge_weight=data_batch.edge_attr,
        batch=data_batch.batch,
        num_nodes=num_nodes,
    )

    assert pooling_out.so.s.size(0) == num_nodes
    assert pooling_out.so.s.size(1) == k
    assert pooling_out.batch.size(0) == num_graphs * k


def test_batch_none(pooler_test_graph_sparse_batch):
    data_batch = pooler_test_graph_sparse_batch

    pooling_out = NMFPooling(k=3).precoarsening(
        edge_index=data_batch.edge_index,
        edge_weight=data_batch.edge_attr,
    )

    # Without an explicit batch vector, precoarsening treats the input as a single graph.
    assert pooling_out.batch.size(0) == pooling_out.so.num_supernodes


if __name__ == "__main__":
    pytest.main([__file__])
