import pytest

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
