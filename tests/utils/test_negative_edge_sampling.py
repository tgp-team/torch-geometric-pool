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
from tgp.utils import batched_negative_edge_sampling, negative_edge_sampling
from tgp.utils.ops import (
    edge_index_to_vector_id,
    sample_almost_k_edges,
    vector_id_to_edge_index,
)


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
    # Create a fully-connected graph:
    N1, N2 = 13, 17
    row = torch.arange(N1).view(-1, 1).repeat(1, N2).view(-1)
    col = torch.arange(N2).view(1, -1).repeat(N1, 1).view(-1)
    edge_index = torch.stack([row, col], dim=0)

    idx = edge_index_to_vector_id(edge_index, (N1, N2))
    assert idx.tolist() == list(range(N1 * N2))
    edge_index2 = torch.stack(vector_id_to_edge_index(idx, (N1, N2)), dim=0)
    assert edge_index.tolist() == edge_index2.tolist()

    vector_id = torch.arange(N1 * N2)
    edge_index3 = torch.stack(vector_id_to_edge_index(vector_id, (N1, N2)), dim=0)
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
    # Use a dense edge set to make prob_neg_edges < _MIN_PROB_EDGES
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
