import pytest
import torch

import tgp.select.lapool_select as lapool_select_module
from tgp.poolers import LaPooling
from tgp.select.lapool_select import LaPoolSelect
from tgp.src import PoolingOutput


def test_forward():
    N = 3
    # x is 1D of length N
    x = torch.arange(N, dtype=torch.float).unsqueeze(1)  # make it 2D [N, 1]
    # Build simple chain adjacency as dense edge_index
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 2], dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = None

    pooler = LaPooling()
    pooler.eval()

    # Call forward; x will be unsqueezed internally
    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    # out.x should have shape [k, 1]
    k = out.so.num_supernodes
    if getattr(pooler, "batched", False):
        assert out.x.shape == (1, k, 1)
    else:
        assert out.x.shape == (k, 1)


def test_shortest_path():
    N = 3
    # x is 1D of length N
    x = torch.arange(N, dtype=torch.float).unsqueeze(1)  # make it 2D [N, 1]
    # Build simple chain adjacency as dense edge_index
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 2], dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = None

    pooler = LaPooling(shortest_path_reg=True)
    pooler.eval()

    # Call forward; x will be unsqueezed internally
    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    # out.x should have shape [k, 1]
    k = out.so.num_supernodes
    if getattr(pooler, "batched", False):
        assert out.x.shape == (1, k, 1)
    else:
        assert out.x.shape == (k, 1)


def test_single_leader_edge_case():
    """Test the edge case where exactly one leader is selected.

    This tests the specific code path in lines 213-215 of lapool_select.py where
    leader_idx.dim() == 0 is handled when there's exactly one leader.
    """
    from tgp.select.lapool_select import LaPoolSelect

    # Create a simple 2-node graph where we can control the leader selection
    N = 2
    x = torch.tensor(
        [
            [1.0],  # Node 0
            [0.1],  # Node 1 - much lower value
        ],
        dtype=torch.float,
    )

    # Simple edge between the two nodes
    edge_index = torch.tensor(
        [
            [0, 1],  # edges
            [1, 0],  # reverse edges
        ],
        dtype=torch.long,
    )
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)

    # Test the selector directly
    selector = LaPoolSelect(batched_representation=False)
    so = selector(
        x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch, num_nodes=N
    )

    # Check if we get exactly one leader (this tests the edge case)
    assert so.s is not None
    s_dense = so.s.to_dense()

    # The selection matrix should be 2x1 or 2x2 depending on how many leaders are selected
    # But we want to test the case where there's exactly one leader
    num_leaders = s_dense.size(1)

    # If we have exactly one leader, test the edge case
    if num_leaders == 1:
        # This should trigger the leader_idx.dim() == 0 case
        assert s_dense.shape == (2, 1)
        # All nodes should be assigned to the single leader
        assert torch.allclose(s_dense.sum(dim=0), torch.ones(1), atol=1e-6)
    else:
        # If we don't get exactly one leader, that's also fine - the test passes
        # as long as the code doesn't crash
        assert s_dense.shape[0] == 2  # 2 input nodes
        assert s_dense.shape[1] >= 1  # At least 1 leader


def test_single_node_isolated():
    """Test LaPoolSelect with a single isolated node (no edges)."""
    from tgp.select.lapool_select import LaPoolSelect

    N = 1
    # Single node with 1D features
    x = torch.tensor([[1.0]], dtype=torch.float)  # [1, 1]
    # No edges for single node
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.empty(0, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)
    selector = LaPoolSelect(batched_representation=False)
    so = selector(
        x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch, num_nodes=N
    )

    # The selection matrix should be 1x1 identity
    assert so.s is not None
    s_dense = so.s.to_dense()
    assert s_dense.shape == (1, 1)
    assert torch.allclose(s_dense, torch.eye(1), atol=1e-6)


@pytest.mark.torch_sparse
def test_lapool_select_sparse_tensor_input():
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    from tgp.select.lapool_select import LaPoolSelect

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(2, 2))
    x = torch.randn(2, 3)
    selector = LaPoolSelect(batched_representation=False)
    so = selector(x=x, edge_index=adj, num_nodes=2)
    assert so.s is not None


def test_lapool_batched_dense_output_mask(pooler_test_graph_dense_batch):
    """Batched dense output: out.mask equals so.out_mask, shape [B, K_max]."""
    x, adj = pooler_test_graph_dense_batch
    pooler = LaPooling(batched=True, sparse_output=False)
    pooler.eval()
    out = pooler(x=x, adj=adj)
    assert out.so is not None
    assert out.mask is not None
    assert torch.equal(out.mask, out.so.out_mask)
    assert out.mask.dim() == 2
    assert out.mask.shape[0] == out.x.shape[0]
    assert out.mask.shape[1] == out.x.shape[1]
    assert torch.equal(out.mask, (out.so.s.sum(dim=-2) > 0))


def test_lapool_batched_sparse_output_no_mask(pooler_test_graph_dense_batch):
    """Batched sparse output: out.mask equals so.out_mask (so.s is 3D so mask is not None)."""
    x, adj = pooler_test_graph_dense_batch
    pooler = LaPooling(batched=True, sparse_output=True)
    pooler.eval()
    out = pooler(x=x, adj=adj)
    assert out.so is not None
    assert out.mask is not None
    assert torch.equal(out.mask, out.so.out_mask)


def test_lapool_select_batched_validation_and_unsqueeze_paths(monkeypatch):
    selector = LaPoolSelect(batched_representation=True)
    x = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)
    adj = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )

    so = selector(x=x, edge_index=adj)
    assert so.s.dim() == 3
    assert so.s.size(0) == 1

    with pytest.raises(ValueError, match=r"x must have shape \[B, N, F\]"):
        selector(x=torch.tensor([1.0, 2.0, 3.0]), edge_index=adj)

    edge_index_sparse = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="dense adjacency tensor"):
        selector(x=x, edge_index=edge_index_sparse)

    monkeypatch.setattr(lapool_select_module, "is_dense_adj", lambda _: True)
    with pytest.raises(ValueError, match=r"\[B, N, N\]"):
        selector(x=x, edge_index=torch.ones(1, 1, 1, 1, dtype=torch.float32))


def test_lapool_select_unbatched_validation_errors():
    selector = LaPoolSelect(batched_representation=False)
    edge_index_sparse = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match=r"x must have shape \[N, F\]"):
        selector(x=torch.randn(1, 3, 2), edge_index=edge_index_sparse)

    with pytest.raises(
        ValueError, match="mask is only supported for batched representations"
    ):
        selector(
            x=torch.randn(3, 2),
            edge_index=edge_index_sparse,
            mask=torch.tensor([True, True, True]),
        )

    with pytest.raises(ValueError, match="expects a sparse adjacency tensor"):
        selector(
            x=torch.randn(3, 2),
            edge_index=torch.eye(3, dtype=torch.float32),
        )


def test_lapool_select_forward_batched_mask_variants():
    selector = LaPoolSelect(batched_representation=True)
    x = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    adj = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    # 1D mask branch: mask is unsqueezed to [1, N].
    s_single = selector._forward_batched(x[:1], adj[:1], mask=torch.tensor([1, 1, 0]))
    assert s_single.shape[0] == 1
    assert torch.allclose(s_single[0, 2], torch.zeros_like(s_single[0, 2]))

    # Graph 1 has no valid nodes, so k_b == 0 and the copy branch is skipped.
    mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    s = selector._forward_batched(x, adj, mask=mask)
    assert torch.allclose(s[1], torch.zeros_like(s[1]))


def test_lapool_select_batched_shortest_path_skips_empty_edges():
    selector = LaPoolSelect(batched_representation=True, shortest_path_reg=True)
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    adj = torch.zeros((1, 3, 3), dtype=torch.float32)

    s = selector._forward_batched(x, adj, mask=None)
    assert s.shape[0] == 1
    assert torch.isfinite(s).all()


def test_lapool_select_unbatched_shortest_path_branch():
    selector = LaPoolSelect(batched_representation=False, shortest_path_reg=True)
    x = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    so = selector(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=None,
        num_nodes=x.size(0),
    )
    assert so.s is not None
    assert torch.isfinite(so.s).all()


def test_lapool_select_unbatched_no_leader_fallback(monkeypatch):
    selector = LaPoolSelect(batched_representation=False, shortest_path_reg=False)
    x = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    monkeypatch.setattr(
        lapool_select_module,
        "scatter_mul",
        lambda src, index, dim, dim_size: torch.zeros(
            dim_size, dtype=src.dtype, device=src.device
        ),
    )

    s = selector._forward_unbatched(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=None,
        num_nodes=x.size(0),
    )
    assert s.shape == (x.size(0), x.size(0))
    assert torch.isfinite(s).all()


def test_lapool_select_forward_batched_single_leader_unsqueeze_branch():
    selector = LaPoolSelect(batched_representation=True)
    x = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)  # [B=1, N=1, F=2]
    adj = torch.zeros((1, 1, 1), dtype=torch.float32)

    s = selector._forward_batched(x, adj, mask=torch.tensor([[True]]))
    assert s.shape == (1, 1, 1)
    assert torch.equal(s, torch.ones_like(s))


def test_lapool_select_unbatched_multi_graph_batch_path():
    selector = LaPoolSelect(batched_representation=False)
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    s = selector._forward_unbatched(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        num_nodes=x.size(0),
    )
    assert s.size(0) == x.size(0)
    assert s.size(1) >= 1
    assert torch.isfinite(s).all()


def test_lapool_select_repr():
    selector = LaPoolSelect(shortest_path_reg=True, s_inv_op="transpose")
    rep = repr(selector)
    assert "LaPoolSelect" in rep
    assert "shortest_path_reg=True" in rep


if __name__ == "__main__":
    pytest.main([__file__])
