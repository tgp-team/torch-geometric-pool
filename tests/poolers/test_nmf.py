import pytest
import torch

from tests.test_utils import make_chain_graph_sparse
from tgp.poolers import NMFPooling
from tgp.select.nmf_select import NMFSelect


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


def test_nmf_pooling_warns_for_unsupported_batched_mode():
    with pytest.warns(
        UserWarning, match="does not support dense padded batched inputs"
    ):
        NMFPooling(k=3, batched=True)


def test_nmf_select_sparse_single_graph_fixed_k():
    x, edge_index, edge_weight, _ = make_chain_graph_sparse(N=4, F_dim=3, seed=42)
    k = 8
    selector = NMFSelect(k=k)
    out = selector(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=x.size(0),
        fixed_k=True,
    )

    assert out.s.dim() == 2
    assert out.s.shape == (x.size(0), k)
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


def test_nmf_select_sparse_batched_with_edgeless_graph():
    # Graph 0: two nodes and one undirected edge.
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # Graph 1: three isolated nodes (no edges).
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    selector = NMFSelect(k=3)
    out = selector(edge_index=edge_index, batch=batch, num_nodes=batch.numel())

    assert out.s.shape == (5, 3)
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


def test_nmf_select_factorize_single_adjacency_empty():
    selector = NMFSelect(k=3)
    adj = torch.zeros((0, 0))

    s = selector._factorize_single_adjacency(adj)

    assert s.shape == (0, 0)


def test_nmf_select_factorize_single_adjacency_actual_k_one():
    selector = NMFSelect(k=1)
    adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    s = selector._factorize_single_adjacency(adj)

    assert s.shape == (2, 1)
    assert torch.allclose(s, torch.ones_like(s))


def test_nmf_select_sparse_batched_with_empty_edge_index_and_empty_graph_slot():
    # Graph ids 0 and 2 are present, graph id 1 has zero nodes.
    batch = torch.tensor([0, 0, 2], dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    selector = NMFSelect(k=2)

    out = selector(edge_index=edge_index, batch=batch, num_nodes=batch.numel())

    assert out.s.shape == (batch.numel(), 2)
    row_sums = out.s.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_nmf_select_repr():
    selector = NMFSelect(k=4, s_inv_op="inverse")

    assert repr(selector) == "NMFSelect(k=4, s_inv_op=inverse)"


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


def test_nmf_lifting_operation(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    pooler = NMFPooling(k=3, batched=False, sparse_output=False)
    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)

    x_lifted = pooler(
        x=out.x,
        so=out.so,
        batch=batch,
        batch_pooled=out.batch,
        lifting=True,
    )

    assert x_lifted.shape == x.shape


def test_nmf_forward_skips_select_when_so_is_provided(
    pooler_test_graph_sparse, monkeypatch
):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    pooler = NMFPooling(k=3, batched=False, sparse_output=False)
    so = pooler.select(
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        num_nodes=x.size(0),
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("select should not be called when so is provided")

    monkeypatch.setattr(pooler, "select", _fail_if_called)
    out = pooler(
        x=x,
        adj=edge_index,
        edge_weight=edge_weight,
        so=so,
        batch=batch,
    )

    assert out.so is so


def test_nmf_extra_repr_args():
    pooler = NMFPooling(k=4, batched=False, cached=True)
    extra = pooler.extra_repr_args()

    assert extra["batched"] is False
    assert extra["cached"] is True


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


def test_nmf_precoarsening_fixed_k_small_graph():
    _, edge_index, edge_weight, _ = make_chain_graph_sparse(N=5, F_dim=3, seed=42)
    k = 8
    pooling_out = NMFPooling(k=k).precoarsening(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=5,
    )

    assert pooling_out.so.s.shape == (5, k)
    assert pooling_out.batch.shape == (k,)


if __name__ == "__main__":
    pytest.main([__file__])
