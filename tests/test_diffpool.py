import pytest
import torch
from torch import Tensor
from torch_sparse import SparseTensor

from tgp.poolers import DiffPool
from tgp.utils.losses import entropy_loss, link_pred_loss


@pytest.fixture(scope="module")
def small_graph_dense():
    """Creates a small dense graph with:
      - Batch size B = 1
      - N = 4 nodes
      - F = 3 features per node
      - Adjacency is a simple symmetric matrix with self-loops
    Returns:
        x: Tensor of shape (B, N, F)
        adj: Tensor of shape (B, N, N).
    """
    B, N, F = 1, 4, 3
    torch.manual_seed(0)
    x = torch.randn((B, N, F), dtype=torch.float)

    # Create adjacency: a complete graph with self-loops
    adj_mat = torch.ones((N, N), dtype=torch.float)
    adj = adj_mat.unsqueeze(0)  # Shape: (1, N, N)
    return x, adj


@pytest.fixture(scope="module")
def small_graph(N=4, F_dim=5):
    row = torch.arange(N - 1, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack(
        [torch.cat([row, col]), torch.cat([col, row])], dim=0
    )  # undirected chain
    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)
    return x, edge_index, edge_weight, batch


@pytest.mark.parametrize("normalize_loss", [False, True])
@pytest.mark.parametrize("link_coeff, ent_coeff", [(1.0, 1.0), (0.5, 2.0)])
def test_diffpool_link_entropy_loss(
    small_graph_dense, normalize_loss, link_coeff, ent_coeff
):
    """Test DiffPool compute_loss under various configurations:
    - normalize_loss False/True
    - different coefficients link_loss_coeff and ent_loss_coeff.
    """
    x, adj = small_graph_dense
    _B, _N, F = x.size(0), x.size(1), x.size(2)
    k = 2  # number of clusters

    # Instantiate DiffPool with specified coefficients and normalize_loss flag
    pooler = DiffPool(
        in_channels=F,
        k=k,
        link_loss_coeff=link_coeff,
        ent_loss_coeff=ent_coeff,
        normalize_loss=normalize_loss,
        remove_self_loops=False,
        degree_norm=False,
        adj_transpose=False,
    )
    pooler.eval()

    # Perform forward pass
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    loss_dict = out.loss

    # Extract assignment matrix S from SelectOutput
    S = out.so.s  # shape (B, N, K)

    # Compute expected link_pred_loss
    expected_link = link_pred_loss(S, adj, normalize_loss=normalize_loss) * link_coeff
    actual_link = loss_dict.get("link_loss", None)
    assert isinstance(actual_link, Tensor)
    assert torch.allclose(actual_link, expected_link, atol=1e-6)

    # Compute expected entropy_loss
    expected_ent = entropy_loss(S) * ent_coeff
    actual_ent = loss_dict.get("entropy_loss", None)
    assert isinstance(actual_ent, Tensor)
    assert torch.allclose(actual_ent, expected_ent, atol=1e-6)


def test_diffpool_edge_cases_and_forward_shapes(small_graph_dense):
    """Test that DiffPool runs without error when:
    - remove_self_loops=True, degree_norm=True, adj_transpose=True
    - Verify shapes of outputs: x_pooled and edge_index.
    """
    x, adj = small_graph_dense
    B, _N, F = x.size(0), x.size(1), x.size(2)
    k = 3  # cluster number can be >1 and <N

    pooler = DiffPool(
        in_channels=F,
        k=k,
        link_loss_coeff=1.0,
        ent_loss_coeff=1.0,
        normalize_loss=True,
        remove_self_loops=True,
        degree_norm=True,
        adj_transpose=True,
    )
    pooler.eval()

    # Forward pass
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)

    # Pooling output should have a loss
    assert out.has_loss
    assert len(out.get_loss_value()) > 0
    assert out.get_loss_value("link_loss") is not None
    assert pooler.has_loss is True

    # x_pooled should have shape (B, k, F)
    x_pooled = out.x
    assert isinstance(x_pooled, Tensor)
    assert x_pooled.shape == (B, k, F)

    # edge_index should be a dense adjacency if adj_transpose=True
    adj_pool = out.edge_index
    assert isinstance(adj_pool, Tensor)
    assert adj_pool.shape == (B, k, k)

    # Ensure no NaNs in pooled adjacency
    assert torch.isfinite(adj_pool).all()

    # Test lifting path: lift back to original nodes
    S_out = out.so
    x_lifted = pooler(x=x_pooled, adj=None, so=S_out, mask=None, lifting=True)
    # Should return Tensor of shape (B, N, F)
    assert isinstance(x_lifted, Tensor)
    assert x_lifted.shape == x.shape

    x_global = pooler.global_pool(x=x, batch=None)
    # Global pooling should return a Tensor of shape (B, F)
    assert isinstance(x_global, Tensor)
    assert x_global.shape == (B, F)


def test_diffpool_invalid_parameters():
    """Test that invalid configurations (e.g., invalid types) raise errors."""
    B, N, F = 1, 4, 3

    # Negative link_loss_coeff is allowed (just scales), so no error
    pooler_neg = DiffPool(in_channels=F, k=2, link_loss_coeff=-1.0, ent_loss_coeff=1.0)
    pooler_neg.eval()
    x = torch.randn((B, N, F))
    adj = torch.ones((B, N, N))
    out = pooler_neg(x=x, adj=adj, so=None, mask=None, lifting=False)
    assert "link_loss" in out.loss

    # Invalid 'in_channels' type (e.g., string) should raise TypeError
    with pytest.raises(TypeError):
        _ = DiffPool(in_channels="invalid", k=2)


def test_diffpool_expressive_cuda_grads(small_graph_dense):
    x, adj = small_graph_dense
    _B, _N, _F = x.size(0), x.size(1), x.size(2)
    k = 2  # number of clusters

    pooler = DiffPool(
        in_channels=3,
        k=k,
        link_loss_coeff=1.0,
        ent_loss_coeff=1.0,
        normalize_loss=True,
        remove_self_loops=False,
        degree_norm=False,
        adj_transpose=False,
        s_inv_op="inverse",
    )
    so = pooler.select(x=x, adj=adj)
    assert so.is_expressive is True, "DiffPool should be expressive"

    # check that so.cuda() fails
    if not torch.cuda.is_available():
        so.to(device=x.device)
        with pytest.raises(AssertionError):
            so.to("cpu").cuda()

    else:
        so.to(device="cuda")
        assert so.s.device.type == "cuda"

    # check if so requires_grad
    assert so.requires_grad_(True).is_expressive is True, (
        "DiffPool should still be expressive with requires_grad=True"
    )


def test_preprocessing_cache(small_graph):
    """Test that preprocessing cache works correctly in DiffPool."""
    x, edge_index, edge_weight, batch = small_graph
    N, F = x.size(0), x.size(1)
    k = 2  # number of clusters

    pooler = DiffPool(
        in_channels=F,
        k=k,
        link_loss_coeff=1.0,
        ent_loss_coeff=1.0,
        normalize_loss=True,
        remove_self_loops=False,
        degree_norm=False,
        adj_transpose=False,
        s_inv_op="inverse",
    )
    pooler.eval()

    ei_sparse = SparseTensor.from_edge_index(
        edge_index, edge_weight, sparse_sizes=(N, N)
    )

    _, adj_p1, _ = pooler.preprocessing(
        x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch, use_cache=True
    )
    _, adj_p2, _ = pooler.preprocessing(
        x=x, edge_index=edge_index, batch=batch, use_cache=True
    )
    _, adj_p3, _ = pooler.preprocessing(
        x=x, edge_index=ei_sparse, batch=batch, use_cache=False
    )

    # Ensure the outputs are the same
    assert adj_p1.equal(adj_p2)
    assert adj_p1.equal(adj_p3)


if __name__ == "__main__":
    pytest.main([__file__])
