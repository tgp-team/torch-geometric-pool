import pytest
import torch
from torch_geometric.nn import DenseGCNConv, GCNConv
from torch_geometric.utils import add_self_loops

from tgp.lift.base_lift import BaseLift, Lift
from tgp.poolers import get_pooler
from tgp.reduce import Reduce
from tgp.select.base_select import SelectOutput


@pytest.fixture(scope="module")
def simple_graph():
    """Create a simple toy graph with 10 nodes, a chain structure, random features, and single-graph batch.

    Returns:
        x           Tensor [N, F] of node features
        edge_index  LongTensor [2, E] of edge indices (undirected)
        edge_weight Tensor [E] of edge weights (all ones)
        batch       LongTensor [N] assigning all nodes to graph 0
    """
    N = 10
    F = 3
    # Chain graph: edges (0-1, 1-2, ..., 8-9), made undirected
    row = torch.arange(9, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    E = edge_index.size(1)

    x = torch.randn((N, F), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_attr=edge_weight, num_nodes=N
    )
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


@pytest.mark.torch_sparse
def test_poolers_forward_and_lifting(simple_graph):
    """For each pooling layer in pooler_map:
    1. Instantiate the pooler with a common set of kwargs
    2. Run preprocessing -> forward (pooling) -> lifting
    3. Verify that outputs have reasonable shapes and types, and no errors occur.
    """
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    x, edge_index, edge_weight, batch = simple_graph
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
    edge_input = edge_index if pooler.is_dense else adj

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


def test_with_tensor(simple_graph):
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

    x, edge_index, edge_weight, batch = simple_graph
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


def test_lift_with_non_tensor_s_inv():
    """Test that BaseLift raises TypeError when s_inv is not a torch.Tensor.

    We use matrix_op="transpose" so that s_inv = so.s, then manually set so.s
    to a non-Tensor value to trigger the check.
    """
    # Create a SelectOutput with valid s
    s = torch.randn((3, 2))
    so = SelectOutput(s=s)

    # Manually set s to something that's not a Tensor (e.g., a list)
    # This bypasses SelectOutput validation and simulates an edge case
    # where s might not be a Tensor (though this shouldn't happen in practice)
    so.s = [1, 2, 3]  # Not a Tensor

    x_pool = torch.randn((2, 3))
    lift = BaseLift(matrix_op="transpose")  # This sets s_inv = so.s

    with pytest.raises(TypeError, match="Expected s_inv to be a torch.Tensor"):
        _ = lift(x_pool, so)


if __name__ == "__main__":
    pytest.main([__file__])
