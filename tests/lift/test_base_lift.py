import pytest
import torch
from torch_geometric.nn import DenseGCNConv, GCNConv

from tgp.lift.base_lift import BaseLift, Lift
from tgp.poolers import get_pooler
from tgp.reduce import Reduce
from tgp.select.base_select import SelectOutput


@pytest.mark.torch_sparse
def test_poolers_forward_and_lifting(pooler_test_graph_sparse):
    """For each pooling layer in pooler_map:
    1. Instantiate the pooler with a common set of kwargs
    2. Run preprocessing -> forward (pooling) -> lifting
    3. Verify that outputs have reasonable shapes and types, and no errors occur.
    """
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    N, F = x.size()

    # Common parameters for all poolers (some may ignore irrelevant keys):
    PARAMS = {
        "in_channels": F,
        "ratio": 0.5,  # select half of the nodes
        "k": max(1, N // 2),  # at least 1 cluster
        "cached": True,
        "lift": "inverse",  # use precomputed inverse
        "s_inv_op": "transpose",
        "lift_red_op": "mean",
        "loss_coeff": 1.0,
        "remove_self_loops": True,
        "scorer": "degree",
        "reduce": "sum",
    }

    # Instantiate the pooler
    pooler = get_pooler("graclus", **PARAMS)
    pooler.eval()

    # Convert edge_index + edge_weight to SparseTensor for pooling
    adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)

    # Choose edge input based on pooler type
    use_batched_dense = pooler.is_dense and getattr(pooler, "batched", False)
    edge_input = edge_index if use_batched_dense else adj

    # 1) Preprocessing: must use 'edge_index=edge_input' to match signature
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=edge_input, x=x, batch=batch, use_cache=False
    )
    assert isinstance(x_pre, torch.Tensor)
    assert isinstance(adj_pre, (SparseTensor, torch.Tensor))
    if mask is not None:
        assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool

    # 2) Forward pass: pooling (signature is x and adj)
    out = pooler(x=x_pre, adj=adj_pre, batch=batch, mask=mask)
    # out should have attributes:
    #   x: pooled features, so: SelectOutput, and optionally edge_index, edge_weight, loss
    assert hasattr(out, "x")
    assert hasattr(out, "so") and isinstance(out.so, SelectOutput)
    assert isinstance(out.x, torch.Tensor)
    num_supernodes = out.so.num_supernodes
    assert 1 <= num_supernodes <= N
    assert 1 <= out.x.size(0) <= N

    # edge_index in out may be a SparseTensor or a Tensor
    ei = out.edge_index
    assert isinstance(ei, (SparseTensor, torch.Tensor))

    # If edge_weight present, check shape consistency
    if hasattr(out, "edge_weight") and out.edge_weight is not None:
        ew = out.edge_weight
        assert isinstance(ew, torch.Tensor)
        if isinstance(ei, torch.Tensor):
            assert ew.numel() == ei.size(1)

    # Apply message passing to ensure output is correct type
    conv = GCNConv(F, F)
    out.x = conv(out.x, out.edge_index)
    assert isinstance(out.x, torch.Tensor)

    # 3) Lifting path: given pooled output out.so and new pooled features x_pool, lift back
    x_pool = out.x.clone()
    x_lifted = pooler(x=x_pool, so=out.so, lifting=True)
    assert isinstance(x_lifted, torch.Tensor)
    assert x_lifted.size(-2) == N
    assert x_lifted.size(-1) == x_pool.size(-1)

    # 4) If there's an auxiliary loss, ensure it is a tensor or float
    if hasattr(out, "loss") and out.loss is not None:
        for loss_val in out.loss.keys():
            # assert isinstance(loss_val, torch.Tensor) or isinstance(loss_val, float)
            assert isinstance(out.loss[loss_val], (torch.Tensor, float)), (
                f"Loss value {loss_val} should be a tensor or float, got {type(out.loss[loss_val])}"
            )


def test_with_tensor(pooler_test_graph_sparse):
    """For each pooling layer in pooler_map:
    1. Instantiate the pooler with a common set of kwargs
    2. Run preprocessing -> forward (pooling) -> lifting
    3. Verify that outputs have reasonable shapes and types, and no errors occur.
    """
    # Try to import SparseTensor for type checking, but don't require it
    try:
        from torch_sparse import SparseTensor

        has_sparse = True
    except ImportError:
        SparseTensor = type(None)  # Dummy type that won't match
        has_sparse = False

    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    N, F = x.size()

    # Common parameters for all poolers (some may ignore irrelevant keys):
    PARAMS = {
        "in_channels": F,
        "ratio": 0.5,  # select half of the nodes
        "k": max(1, N // 2),  # at least 1 cluster
        "cached": True,
        "lift": "inverse",  # use precomputed inverse
        "s_inv_op": "inverse",
        "lift_red_op": "mean",
        "loss_coeff": 1.0,
        "remove_self_loops": True,
        "scorer": "degree",
        "reduce": "sum",
    }

    # Instantiate the pooler
    pooler = get_pooler("mincut", **PARAMS)
    pooler.eval()

    edge_input = edge_index

    # 1) Preprocessing: must use 'edge_index=edge_input' to match signature
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=edge_input, x=x, batch=batch, use_cache=False
    )
    assert isinstance(x_pre, torch.Tensor)
    if has_sparse:
        assert isinstance(adj_pre, (SparseTensor, torch.Tensor))
    else:
        assert isinstance(adj_pre, torch.Tensor)
    if mask is not None:
        assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool

    # 2) Forward pass: pooling (signature is x and adj)
    out = pooler(x=x_pre, adj=adj_pre, batch=batch, mask=mask)
    # out should have attributes:
    #   x: pooled features, so: SelectOutput, and optionally edge_index, edge_weight, loss
    assert hasattr(out, "x")
    assert hasattr(out, "so") and isinstance(out.so, SelectOutput)
    assert isinstance(out.x, torch.Tensor)
    num_supernodes = out.so.num_supernodes
    assert 1 <= num_supernodes <= N
    assert 1 <= out.x.size(0) <= N

    # edge_index in out may be a SparseTensor or a Tensor
    ei = out.edge_index
    if has_sparse:
        assert isinstance(ei, (SparseTensor, torch.Tensor))
    else:
        assert isinstance(ei, torch.Tensor)

    # If edge_weight present, check shape consistency
    if hasattr(out, "edge_weight") and out.edge_weight is not None:
        ew = out.edge_weight
        assert isinstance(ew, torch.Tensor)
        if isinstance(ei, torch.Tensor):
            assert ew.numel() == ei.size(1)

    # Apply message passing to ensure output is correct type
    conv = DenseGCNConv(F, F)
    out.x = conv(out.x, out.edge_index)
    assert isinstance(out.x, torch.Tensor)

    # 3) Lifting path: given pooled output out.so and new pooled features x_pool, lift back
    x_pool = out.x.clone()
    x_lifted = pooler(x=x_pool, so=out.so, lifting=True)
    assert isinstance(x_lifted, torch.Tensor)
    assert x_lifted.size(-2) == N
    assert x_lifted.size(-1) == x_pool.size(-1)

    # 4) If there's an auxiliary loss, ensure it is a tensor or float
    if hasattr(out, "loss") and out.loss is not None:
        for loss_val in out.loss.keys():
            # assert isinstance(loss_val, torch.Tensor) or isinstance(loss_val, float)
            assert isinstance(out.loss[loss_val], (torch.Tensor, float)), (
                f"Loss value {loss_val} should be a tensor or float, got {type(out.loss[loss_val])}"
            )


def test_lift_repr():
    lift = Lift()
    assert len(repr(lift)) > 0


def test_reduce_repr():
    reduce = Reduce()
    assert len(repr(reduce)) > 0


def test_invalid_lift_op():
    with pytest.raises(RuntimeError):
        # Attempt to create a BaseLift with an invalid lift operation
        s = torch.randn((3, 3))
        so = SelectOutput(s)
        x = torch.randn((3, 2))
        BaseLift(matrix_op="invalid_op")(x, so)


def test_lift_dense_multi_graph_raises_on_inconsistent_blocks():
    lift_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float)
    x_pool_flat = torch.randn(6, 2)
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    batch_pooled = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="Inconsistent per-graph blocks"):
        BaseLift._lift_dense_multi_graph(lift_matrix, x_pool_flat, batch, batch_pooled)


def test_forward_dense_unbatched_multi_graph_with_global_pooled_features():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float
    )
    so = SelectOutput(s=lift_matrix, batch=torch.tensor([0, 0, 1, 1], dtype=torch.long))
    x_pool = torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float)

    out = BaseLift(matrix_op="transpose")(x_pool, so)
    expected = lift_matrix.matmul(x_pool)
    torch.testing.assert_close(out, expected)


def test_forward_dense_unbatched_multi_graph_raises_on_invalid_pooled_rows():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float
    )
    so = SelectOutput(s=lift_matrix, batch=torch.tensor([0, 0, 1, 1], dtype=torch.long))
    x_pool = torch.randn(3, 2)  # Neither K (=2) nor B*K (=4).

    with pytest.raises(ValueError, match="Unexpected pooled feature shape"):
        BaseLift(matrix_op="transpose")(x_pool, so)


def test_forward_dense_unbatched_multi_graph_raises_on_bad_batch_pooled_length():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float
    )
    so = SelectOutput(s=lift_matrix, batch=torch.tensor([0, 0, 1, 1], dtype=torch.long))
    x_pool = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=torch.float
    )

    with pytest.raises(ValueError, match="batch_pooled has an unexpected length"):
        BaseLift(matrix_op="transpose")(
            x_pool, so, batch_pooled=torch.tensor([0, 0, 1], dtype=torch.long)
        )


def test_forward_dense_unbatched_multi_graph_with_default_batch_pooled():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    so = SelectOutput(s=lift_matrix, batch=batch)
    x_pool = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=torch.float
    )

    out = BaseLift(matrix_op="transpose")(x_pool, so)
    expected = torch.cat(
        [lift_matrix[:2].matmul(x_pool[:2]), lift_matrix[2:].matmul(x_pool[2:])], dim=0
    )
    torch.testing.assert_close(out, expected)


def test_forward_dense_unbatched_with_3d_pooled_features_bad_batch_pooled_length():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float
    )
    so = SelectOutput(s=lift_matrix, batch=torch.tensor([0, 0, 1, 1], dtype=torch.long))
    x_pool = torch.randn(2, 2, 3)  # [B, K, F]

    with pytest.raises(ValueError, match="batch_pooled has an unexpected length"):
        BaseLift(matrix_op="transpose")(
            x_pool, so, batch_pooled=torch.tensor([0, 0, 1], dtype=torch.long)
        )


def test_forward_dense_batched_assignment_expands_compacted_rows():
    s = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float,
    )  # [B=2, N=3, K=3]
    so = SelectOutput(s=s)
    x_pool_compact = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float)

    out = BaseLift(matrix_op="transpose")(x_pool_compact, so)

    expected_full = torch.tensor(
        [[10.0], [0.0], [20.0], [0.0], [30.0], [0.0]], dtype=torch.float
    ).view(2, 3, 1)
    expected = s.matmul(expected_full)
    torch.testing.assert_close(out, expected)


def test_forward_uses_precomputed_inverse_matrix():
    lift_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float)
    so = SelectOutput(s=lift_matrix, s_inv=lift_matrix.transpose(-2, -1))
    x_pool = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float)

    out = BaseLift(matrix_op="precomputed")(x_pool, so)
    torch.testing.assert_close(out, lift_matrix.matmul(x_pool))


def test_forward_dense_unbatched_with_3d_pooled_features_single_graph():
    lift_matrix = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float)
    so = SelectOutput(s=lift_matrix, batch=torch.zeros(2, dtype=torch.long))
    x_pool = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]], dtype=torch.float)  # [1, K, F]

    out = BaseLift(matrix_op="transpose")(x_pool, so)
    expected = lift_matrix.matmul(x_pool.squeeze(0))
    torch.testing.assert_close(out, expected)


def test_forward_dense_unbatched_with_3d_pooled_features_multi_graph_default_batch():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    so = SelectOutput(s=lift_matrix, batch=batch)
    x_pool = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]], dtype=torch.float
    )  # [B=2, K=2, F=2]

    out = BaseLift(matrix_op="transpose")(x_pool, so)
    expected = torch.cat(
        [lift_matrix[:2].matmul(x_pool[0]), lift_matrix[2:].matmul(x_pool[1])], dim=0
    )
    torch.testing.assert_close(out, expected)


def test_forward_dense_unbatched_multi_graph_with_provided_batch_pooled():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    so = SelectOutput(s=lift_matrix, batch=batch)
    x_pool = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=torch.float
    )
    batch_pooled = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = BaseLift(matrix_op="transpose")(x_pool, so, batch_pooled=batch_pooled)
    expected = torch.cat(
        [
            lift_matrix[:2].matmul(x_pool[:2]),
            lift_matrix[2:].matmul(x_pool[2:]),
        ],
        dim=0,
    )
    torch.testing.assert_close(out, expected)


def test_forward_dense_unbatched_with_3d_pooled_features_multi_graph_provided_batch():
    lift_matrix = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    so = SelectOutput(s=lift_matrix, batch=batch)
    x_pool = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]], dtype=torch.float
    )  # [B=2, K=2, F=2]
    batch_pooled = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = BaseLift(matrix_op="transpose")(x_pool, so, batch_pooled=batch_pooled)
    expected = torch.cat(
        [lift_matrix[:2].matmul(x_pool[0]), lift_matrix[2:].matmul(x_pool[1])], dim=0
    )
    torch.testing.assert_close(out, expected)


def test_forward_dense_batched_assignment_without_expansion():
    s = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float,
    )  # [B=2, N=3, K=2]
    so = SelectOutput(s=s)
    x_pool = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float)

    out = BaseLift(matrix_op="transpose")(x_pool, so)
    expected = s.matmul(x_pool.view(2, 2, 1))
    torch.testing.assert_close(out, expected)


def test_base_lift_repr():
    lift = BaseLift(matrix_op="transpose", reduce_op="mean")
    repr_str = repr(lift)
    assert "BaseLift" in repr_str
    assert "matrix_op=transpose" in repr_str
    assert "reduce_op=mean" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
