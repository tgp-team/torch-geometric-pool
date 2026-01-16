import random
from itertools import product

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import erdos_renyi_graph
from torch_sparse import SparseTensor

from tgp.poolers.bnpool_sparse import SparseBNPool
from tgp.select import DPSelect


@pytest.fixture
def set_random_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(params=list(product([3, 5, 20], [100, 200, 500])))
# @pytest.fixture(params=list(product([3], [100])))
def small_batched_sparse_graphs(set_random_seed, request):
    batch_size, max_num_nomes = request.param
    n_features = 5
    min_p = 0.8
    num_nodes_per_graph = [
        round((min_p + random.random() * (1 - min_p)) * max_num_nomes)
        for _ in range(batch_size)
    ]
    graph_list = [
        Data(
            x=torch.randn((n, n_features)),
            edge_index=erdos_renyi_graph(n, edge_prob=0.2),
        )
        for n in num_nodes_per_graph
    ]
    batched_data = Batch.from_data_list(graph_list)
    return batched_data


@pytest.fixture(params=list(product([100, 200], [0.1, 0.2])))
def single_sparse_graph(set_random_seed, request):
    n_nodes, p = request.param
    n_features = 3
    return Data(
        x=torch.randn((n_nodes, n_features)), edge_index=erdos_renyi_graph(n_nodes, p)
    )


def test_sparse_bnpool_initialization():
    """Test BNPool initialization with different parameters."""
    # Test valid initialization
    pooler = SparseBNPool(in_channels=4, k=3)
    assert pooler.k == 3
    assert pooler.alpha_DP == 1.0

    # Test custom parameters
    pooler = SparseBNPool(
        in_channels=4, k=3, alpha_DP=2.0, K_var=0.5, K_mu=5.0, K_init=0.5, eta=0.8
    )
    assert pooler.alpha_DP == 2.0
    assert pooler.K_var_val == 0.5
    assert pooler.K_mu_val == 5.0
    assert pooler.K_init_val == 0.5
    assert pooler.eta == 0.8


def test_sparse_bnpool_invalid_parameters():
    """Test BNPool initialization with invalid parameters."""
    with pytest.raises(ValueError, match="alpha_DP must be positive"):
        SparseBNPool(in_channels=4, k=3, alpha_DP=-1.0)

    with pytest.raises(ValueError, match="K_var must be positive"):
        SparseBNPool(in_channels=4, k=3, K_var=-1.0)

    with pytest.raises(ValueError, match="eta must be positive"):
        SparseBNPool(in_channels=4, k=3, eta=-1.0)

    with pytest.raises(ValueError, match="max_k must be positive"):
        SparseBNPool(in_channels=4, k=-3)


@pytest.mark.parametrize("train_k", [True, False])
def test_sparse_bnpool_training_mode_on_batch(small_batched_sparse_graphs, train_k):
    """Test BNPool behavior in training mode."""
    batched_graphs = small_batched_sparse_graphs
    x, edge_index, batch = (
        batched_graphs.x,
        batched_graphs.edge_index,
        batched_graphs.batch,
    )

    pooler = SparseBNPool(in_channels=x.shape[-1], k=3, train_K=train_k)
    pooler.train()

    # Enable gradient tracking
    x.requires_grad_(True)

    # Forward pass
    out = pooler(x=x, adj=edge_index, batch=batch)

    # Check if loss components are present
    assert isinstance(out.loss, dict)
    assert "quality" in out.loss
    assert "kl" in out.loss
    assert "K_prior" in out.loss

    # Check if losses are differentiable
    total_loss = sum(out.loss.values())
    total_loss.backward()

    # Check if gradients are computed
    assert x.grad is not None
    if pooler.train_K:
        assert pooler.K.grad is not None
    else:
        assert pooler.K.grad is None


@pytest.mark.parametrize("train_k", [True, False])
def test_sparse_bnpool_training_modeon_single_graph(single_sparse_graph, train_k):
    """Test BNPool behavior in training mode."""
    s_graph = single_sparse_graph
    x, edge_index = s_graph.x, s_graph.edge_index

    pooler = SparseBNPool(in_channels=x.shape[-1], k=3, train_K=train_k)
    pooler.train()

    # Enable gradient tracking
    x.requires_grad_(True)

    # Forward pass
    out = pooler(x=x, adj=edge_index)

    # Check if loss components are present
    assert isinstance(out.loss, dict)
    assert "quality" in out.loss
    assert "kl" in out.loss
    assert "K_prior" in out.loss

    # Check if losses are differentiable
    total_loss = sum(out.loss.values())
    total_loss.backward()

    # Check if gradients are computed
    assert x.grad is not None
    if pooler.train_K:
        assert pooler.K.grad is not None
    else:
        assert pooler.K.grad is None


@pytest.mark.parametrize("train_k", [True, False])
def test_sparse_bnpool_single_graph_with_batch_vector(single_sparse_graph, train_k):
    """Test BNPool with a single graph that has a batch vector (all values same).

    This tests the behavior where DPSelect with batched_representation=False uses the dense
    [N, K] representation for single-graph batches, which is useful when DataLoaders produce
    single-graph batches (e.g., last batch when batch_size * N + 1 graphs).
    """
    s_graph = single_sparse_graph
    x, edge_index = s_graph.x, s_graph.edge_index
    N = x.size(0)

    # Create a batch vector where all values are the same (single graph in batch format)
    batch = torch.zeros(N, dtype=torch.long)

    pooler = SparseBNPool(in_channels=x.shape[-1], k=3, train_K=train_k)
    pooler.train()

    # Enable gradient tracking
    x.requires_grad_(True)

    # Forward pass with batch vector (should use dense path automatically)
    out = pooler(x=x, adj=edge_index, batch=batch)

    # Check if loss components are present
    assert isinstance(out.loss, dict)
    assert "quality" in out.loss
    assert "kl" in out.loss
    assert "K_prior" in out.loss

    # Check output shapes - should be k supernodes (not batch_size * k)
    k = pooler.k
    assert out.x.shape[0] == k, (
        "For single graph with batch vector, should have k supernodes (not batch_size * k)"
    )
    assert out.x.shape[1] == x.shape[1]

    # Check if losses are differentiable
    total_loss = sum(out.loss.values())
    total_loss.backward()

    # Check if gradients are computed
    assert x.grad is not None
    if pooler.train_K:
        assert pooler.K.grad is not None
    else:
        assert pooler.K.grad is None


def test_sparse_bnpool_eval_mode(small_batched_sparse_graphs):
    batched_graphs = small_batched_sparse_graphs
    x, edge_index, batch = (
        batched_graphs.x,
        batched_graphs.edge_index,
        batched_graphs.batch,
    )
    batch_size = batch.max().item() + 1
    _, n_features = x.shape

    k = 2
    pooler = SparseBNPool(in_channels=x.shape[-1], k=k)
    pooler.eval()

    out = pooler(x=x, adj=edge_index, batch=batch)
    # TODO: how should be the size of the output?
    # Check output shapes
    assert out.x.shape[0] == batch_size * k, (
        "N nodes should be k*batch_size for pooled x"
    )
    assert out.x.shape[1] == n_features, "Number of nodes should be equal to k"

    assert out.edge_index.ndim == 2, "Edge index should be a sparse adjacency matrix"

    assert out.edge_index.shape[0] == 2, "First dimension should be 2 for edge_index"
    assert out.edge_index.shape[1] == batch_size * k, (
        "Adjacency matrix sparse size should match batch_size * k"
    )


def test_dpselect_sparse_multi_graph_returns_dense():
    """Test that DPSelect with batched_representation=False returns a dense [N, K] tensor for multi-graph batches.

    The dense representation is more memory-efficient and faster to compute
    compared to block-diagonal sparse representation.
    """
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    selector = DPSelect(in_channels=4, k=3, batched_representation=False)
    so = selector(x=x, batch=batch)

    # Returns dense [N, K] tensor instead of block-diagonal SparseTensor [N, B*K]
    assert isinstance(so.s, torch.Tensor)
    assert not isinstance(so.s, SparseTensor)
    assert so.s.shape == (6, 3)  # [N, K] not [N, B*K]


# def test_dpselect_sparse_empty_batch_returns_dense():
#     """Test DPSelect with batched_representation=False with empty batch returns dense tensor."""
#     x = torch.empty((0, 4))
#     batch = torch.empty((0,), dtype=torch.long)
#     selector = DPSelect(in_channels=4, k=3, batched_representation=False)
#     so = selector(x=x, batch=batch)

#     # Returns dense [N, K] tensor
#     assert isinstance(so.s, torch.Tensor)
#     assert so.s.shape[0] == 0  # Empty tensor
#     assert so.s.shape[1] == 3  # K clusters


def test_sparse_bnpool_lifting_operation(small_batched_sparse_graphs):
    """Test the lifting operation in BNPool."""
    # TODO: to do
    pass
