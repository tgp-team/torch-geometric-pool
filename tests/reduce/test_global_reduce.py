import pytest
import torch

from tgp.reduce import GlobalReduce
from tgp.reduce.global_reduce import _validate_dense_mask


@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max", "min"])
def test_readout_dense_all_ops(reduce_op):
    # Dense: x [B=2, N=3, F=2] -> readout infers dense
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # graph 0
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],  # graph 1
        ],
        dtype=torch.float,
    )
    reducer = GlobalReduce(reduce_op=reduce_op)
    out = reducer(x)
    assert out.shape == (2, 2)  # [B, F]

    if reduce_op == "sum":
        expected0 = torch.tensor([1.0 + 3.0 + 5.0, 2.0 + 4.0 + 6.0])
        expected1 = torch.tensor([-1.0 + 0.0 + 2.0, 0.0 + 1.0 + (-2.0)])
    elif reduce_op == "mean":
        expected0 = torch.tensor([(1.0 + 3.0 + 5.0) / 3.0, (2.0 + 4.0 + 6.0) / 3.0])
        expected1 = torch.tensor([(-1.0 + 0.0 + 2.0) / 3.0, (0.0 + 1.0 + (-2.0)) / 3.0])
    elif reduce_op == "max":
        expected0 = torch.tensor([5.0, 6.0])
        expected1 = torch.tensor([2.0, 1.0])
    else:  # min
        expected0 = torch.tensor([1.0, 2.0])
        expected1 = torch.tensor([-1.0, -2.0])

    expected = torch.stack([expected0, expected1], dim=0)
    assert torch.equal(out, expected)

    with pytest.raises(ValueError, match="Unknown aggregator alias|invalid"):
        GlobalReduce(reduce_op="invalid")


def test_readout_sparse():
    # Sparse: x [N=4, F=2], batch -> readout infers sparse
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    reducer = GlobalReduce(reduce_op="sum")
    out = reducer(x, batch=batch)
    assert out.shape == (2, 2)  # [B, F]


def test_readout_dense_mask_all_false_preserves_batch_size():
    x = torch.randn(3, 4, 2)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    reducer = GlobalReduce(reduce_op="sum")
    out = reducer(x, mask=mask)
    assert out.shape == (3, 2)
    assert torch.allclose(out, torch.zeros_like(out))


def test_readout_sparse_single_graph():
    # Sparse with batch=None -> single graph, output [1, F]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    reducer = GlobalReduce(reduce_op="sum")
    out = reducer(x, batch=None)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[9.0, 12.0]]))


def test_readout_sparse_with_size():
    # Sparse with explicit size
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    reducer = GlobalReduce(reduce_op="sum")
    out = reducer(x, batch=batch, size=2)
    assert out.shape == (2, 2)


def test_readout_sparse_size_requires_batch():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
    with pytest.raises(
        ValueError,
        match="size is only supported for sparse readout when batch is provided",
    ):
        GlobalReduce(reduce_op="sum")(x, batch=None, size=2)


def test_readout_sparse_rejects_mask():
    # mask is only valid for dense [B, N, F] inputs.
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    mask = torch.tensor([True, True, False, True], dtype=torch.bool)
    reducer = GlobalReduce(reduce_op="sum")
    with pytest.raises(ValueError, match="mask is only supported for dense"):
        reducer(x, batch=batch, mask=mask)


def test_readout_dense_rejects_wrong_shape_mask():
    # dense readout requires mask shape [B, N] exactly.
    x = torch.randn(2, 3, 4)  # [B=2, N=3, F=4]
    mask_1d = torch.ones(6, dtype=torch.bool)  # wrong: 1D
    reducer = GlobalReduce(reduce_op="sum")
    with pytest.raises(ValueError, match="mask must have shape \\[B, N\\]"):
        reducer(x, mask=mask_1d)
    mask_wrong = torch.ones(2, 5, dtype=torch.bool)  # wrong: N=5 != 3
    with pytest.raises(ValueError, match="mask must have shape \\[B, N\\]"):
        reducer(x, mask=mask_wrong)


def test_readout_invalid_ndim():
    x = torch.randn(2, 3, 4, 5)  # 4D
    reducer = GlobalReduce(reduce_op="sum")
    with pytest.raises(ValueError, match="2D.*3D"):
        reducer(x)


@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max", "min"])
def test_readout_dense_with_mask(reduce_op):
    # Dense with mask: [B=2, N=3, F=2], mask zeros out one node per graph
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],
        ],
        dtype=torch.float,
    )
    mask = torch.tensor(
        [[True, True, False], [True, False, True]], dtype=torch.bool
    )  # (2, 3)
    reducer = GlobalReduce(reduce_op=reduce_op)
    out = reducer(x, mask=mask)
    assert out.shape == (2, 2)
    if reduce_op == "sum":
        # Graph 0: nodes 0,1 -> [4, 6]; Graph 1: nodes 0,2 -> [1, -2]
        assert torch.allclose(out, torch.tensor([[4.0, 6.0], [1.0, -2.0]]))
    elif reduce_op == "mean":
        # Graph 0: mean of 2 nodes [2, 3]; Graph 1: mean of 2 nodes [0.5, -1]
        assert torch.allclose(out, torch.tensor([[2.0, 3.0], [0.5, -1.0]]))
    elif reduce_op == "max":
        # Graph 0: max over nodes 0,1 -> [3, 4]; Graph 1: max over nodes 0,2 -> [2, 0]
        assert torch.allclose(out, torch.tensor([[3.0, 4.0], [2.0, 0.0]]))
    else:  # min
        # Graph 0: min over nodes 0,1 -> [1, 2]; Graph 1: min over nodes 0,2 -> [-1, -2]
        assert torch.allclose(out, torch.tensor([[1.0, 2.0], [-1.0, -2.0]]))


@pytest.mark.parametrize("reduce_op", ["sum", "mean"])
def test_readout_pyg_aggr(reduce_op):
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    if reduce_op == "sum":
        aggr_module = aggr.SumAggregation()
    else:
        aggr_module = aggr.MeanAggregation()
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    reducer_module = GlobalReduce(reduce_op=aggr_module)
    out = reducer_module(x, batch=batch)
    assert out.shape == (2, 2)
    # Compare to string path
    reducer_str = GlobalReduce(reduce_op=reduce_op)
    out_str = reducer_str(x, batch=batch)
    assert torch.allclose(out, out_str)


def test_globalreduce_trainable_aggregator_parameters_registered():
    """Trainable aggregators (e.g. LSTM) should be registered as module params."""
    try:
        from torch_geometric.nn import aggr  # noqa: F401
    except ImportError:
        pytest.skip("PyG aggr not available")

    # LSTMAggregation may not exist depending on PyG version; skip in that case.
    try:
        readout_module = GlobalReduce(reduce_op="lstm", in_channels=4)
    except (ValueError, TypeError):
        pytest.skip("LSTMAggregation not available in this PyG version")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = readout_module

    model = Model()
    sub_params = {id(p) for p in model.readout.parameters()}
    model_params = {id(p) for p in model.parameters()}
    assert sub_params.issubset(model_params)


def test_readout_pyg_aggr_sparse_single_graph():
    # PyG aggr with batch=None (single graph)
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
    reducer = GlobalReduce(reduce_op=aggr.SumAggregation())
    out = reducer(x, batch=None)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[4.0, 6.0]]))


def test_readout_pyg_aggr_sparse_rejects_mask():
    """Readout rejects mask for sparse (2D) inputs."""
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    reducer = GlobalReduce(reduce_op=aggr.SumAggregation())
    with pytest.raises(ValueError, match="mask is only supported for dense"):
        reducer(x, batch=batch, mask=mask)


def test_readout_pyg_aggr_dense_with_mask():
    # Dense + PyG aggr + mask
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],
        ],
        dtype=torch.float,
    )
    mask = torch.tensor([[True, True, False], [False, True, True]], dtype=torch.bool)
    reducer = GlobalReduce(reduce_op=aggr.MeanAggregation())
    out = reducer(x, mask=mask)
    assert out.shape == (2, 2)
    # Graph 0: mean of nodes 0,1 -> [2, 3]; Graph 1: mean of nodes 1,2 -> [1, -0.5]
    expected = torch.tensor([[2.0, 3.0], [1.0, -0.5]])
    assert torch.allclose(out, expected)


def test_validate_dense_mask_accepts_none():
    x = torch.randn(2, 3, 4)
    # Private helper no-op path: should just return when mask is None.
    assert _validate_dense_mask(None, x) is None
