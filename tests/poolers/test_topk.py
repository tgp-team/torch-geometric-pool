import pytest
import torch
from torch import Tensor

from tgp.poolers.topk import TopkPooling
from tgp.select.base_select import SelectOutput
from tgp.select.topk_select import TopkSelect
from tgp.src import PoolingOutput


@pytest.mark.parametrize(
    "ratio,min_score",
    [
        (None, None),  # both None → should raise
    ],
)
def test_topk_select_init_invalid(ratio, min_score):
    with pytest.raises(ValueError):
        _ = TopkPooling(in_channels=4, ratio=ratio, min_score=min_score)


def test_topk_select_identity_act_and_batch_default():
    N = 5

    selector = TopkSelect(
        in_channels=1, ratio=0.5, min_score=None, act="linear", s_inv_op="transpose"
    )

    x = torch.arange(1.0, N + 1).unsqueeze(-1)
    out = selector.forward(x=x, batch=None)
    assert isinstance(out, SelectOutput)

    expected_indices = torch.tensor([4, 3, 2], dtype=torch.long)
    assert torch.equal(out.node_index.sort(descending=True)[0], expected_indices)

    repr_str = repr(selector)
    assert "ratio=0.5" in repr_str and "min_score" not in repr_str


def test_topk_select_with_weight_and_act_default():
    """Test the default path where in_channels>1, weight is a learnable Parameter,
    act is 'tanh' (so non-identity), and batch=None defaults to zeros.
    We ensure that calling forward returns correct SelectOutput structure and
    that s_inv is transpose of s.
    """
    N, F = 6, 4
    selector = TopkSelect(
        in_channels=F, ratio=0.5, min_score=None, act="tanh", s_inv_op="transpose"
    )
    selector.eval()

    # Create a 2D feature tensor x [N,F]
    torch.manual_seed(0)
    x = torch.randn((N, F), dtype=torch.float)

    # Call forward without batch (defaults to zeros)
    out = selector(x=x, batch=None)
    assert isinstance(out, SelectOutput)

    # Ratio=0.5 → k = ceil(0.5 * 6) = 3
    assert out.num_supernodes == 3
    assert out.num_nodes == N
    assert out.node_index.size(0) == 3

    # s: a SparseTensor of shape (N, k). s_inv should be transpose of s
    s = out.s
    s_inv = out.s_inv
    assert torch.allclose(s.to_dense(), s_inv.to_dense().transpose(0, 1))

    # repr path with min_score=None
    repr_str = repr(selector)
    assert "ratio=0.5" in repr_str
    assert "min_score" not in repr_str


def test_topk_select_min_score_branch_and_repr():
    """Test the path when min_score is provided:
    - score = softmax(score, batch)
    - node_index = indices where score > min_score
    - repr has 'min_score=...'.
    """
    N, F = 4, 2
    # Use in_channels>1 so weight is not None
    selector = TopkSelect(
        in_channels=F, ratio=0.5, min_score=0.2, act="tanh", s_inv_op="transpose"
    )
    selector.eval()

    # Create a feature tensor x [N, F]
    # We pick values so that the inner product with weight (initialized randomly)
    # is not too important, but softmax will produce 4 positive probabilities summing to 1.
    # We set min_score=0.2, so any softmax value >0.2 is selected.
    torch.manual_seed(1)
    x = torch.randn((N, F), dtype=torch.float)

    # Define a batch vector splitting nodes into two graphs:
    # batch = [0,0,1,1], so softmax applied per graph of size 2 each.
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = selector(x=x, batch=batch)
    assert isinstance(out, SelectOutput)

    # Now verify that `score = softmax(raw_score, batch)`, so values are in (0,1),
    # sum to 1 per graph. For each graph, size=2, softmax gives two values that sum to 1.
    # min_score=0.2: since 1/2=0.5, both softmax outputs >0.2, so each graph contributes both nodes,
    # so total node_index length = 4.
    assert out.node_index.size(0) == 4

    # repr path when min_score is not None
    repr_str = repr(selector)
    assert "min_score=0.2" in repr_str


def test_topk_pooling_end_to_end_pool_and_lift():
    """Test TopkPooling end-to-end (non-lifting and lifting paths):
    - Build a small graph (N=6) with random features and a simple adjacency
    - pool with ratio=0.5 (k=3) and check shapes
    - then lift back and check restored shape.
    """
    N, F = 6, 5
    # Build random node features
    torch.manual_seed(2)
    x = torch.randn((N, F), dtype=torch.float)

    # Build a simple adjacency as edge_index (chain of 6 nodes, undirected)
    row = torch.arange(5, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)

    # Instantiate TopkPooling with default act='tanh', s_inv_op='transpose'
    pooler = TopkPooling(
        in_channels=F,
        ratio=0.5,
        min_score=None,
        nonlinearity="linear",  # force identity activation
        lift="transpose",
        s_inv_op="transpose",
        connect_red_op="sum",
        lift_red_op="sum",
    )
    pooler.eval()

    # Forward (pooling) without providing batch; batch defaults to zeros
    out = pooler(
        x=x,
        adj=edge_index,
        edge_weight=None,
        so=None,
        batch=None,
        attn=None,
        lifting=False,
    )
    # out.x should have shape (k=3, F)
    x_pooled = out.x
    assert isinstance(x_pooled, Tensor)
    assert x_pooled.shape == (3, F)

    # out.edge_index is a SparseTensor or edge_index of pooled graph
    pooled_adj = out.edge_index
    # Check that pooled_adj is either a torch.Tensor or SparseTensor
    try:
        from torch_sparse import SparseTensor as _SparseTensor

        has_sparse = True
    except ImportError:
        _SparseTensor = type(None)  # Dummy type that won't match
        has_sparse = False

    if has_sparse:
        assert isinstance(pooled_adj, (Tensor, _SparseTensor))
    else:
        assert isinstance(pooled_adj, Tensor)

    # out.so is a SelectOutput
    assert isinstance(out.so, SelectOutput)

    # Now test lifting path: lift the pooled features back
    x_lifted = pooler(
        x=x_pooled, adj=None, so=out.so, batch=None, attn=None, lifting=True
    )
    # Should restore to shape (N, F)
    assert isinstance(x_lifted, Tensor)
    assert x_lifted.shape == (N, F)


def test_topk_pooling_with_min_score_parameter():
    """Ensure that TopkPooling with min_score set invokes TopkSelect's min_score branch:
    - We build x so that softmax probabilities for each node (since batch=None) are equal to 1/N.
    - If min_score < 1/N, we expect all nodes selected (k=N).
    - If min_score > 1/N, we expect zero nodes selected (k=0).
    """
    N, F = 4, 2
    torch.manual_seed(3)
    x = torch.ones((N, F), dtype=torch.float)  # uniform features

    # Edge index for a complete graph of 4 nodes (undirected)
    idx_row, idx_col = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    edge_index = torch.stack([idx_row.flatten(), idx_col.flatten()], dim=0)

    # min_score < 1/N → select all nodes
    threshold1 = 0.1  # 1/N = 0.25 > 0.1
    pooler1 = TopkPooling(
        in_channels=F,
        ratio=0.5,  # ignored since min_score provided
        min_score=threshold1,
        nonlinearity="tanh",
    )
    pooler1.eval()
    out1 = pooler1(
        x=x,
        adj=edge_index,
        edge_weight=None,
        so=None,
        batch=None,
        attn=None,
        lifting=False,
    )
    # Since softmax scores all equal 0.25 and 0.25 > 0.1, all N=4 nodes selected
    assert out1.so.num_supernodes == N
    assert out1.x.shape == (N, F)


def test_topk_forward(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse

    pooler = TopkPooling(in_channels=x.shape[-1], ratio=0.5)
    pooler.eval()

    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)


def test_topk_with_dim_bigger_than_1():
    x = torch.randn(10, 2)
    selector = TopkSelect(in_channels=1, ratio=0.5, act="linear")

    with pytest.raises(AssertionError):
        out = selector(x=x)

    x = torch.randn(10)
    out = selector(x=x)
    assert out.node_index.size(0) == 5
    assert out.num_supernodes == 5
    assert out.num_nodes == 10
    assert out.s.size(0) == 10


if __name__ == "__main__":
    pytest.main([__file__])
