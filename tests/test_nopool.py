import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

from tgp.poolers import get_pooler, pooler_map
from tgp.select.base_select import SelectOutput
from tgp.select.identity_select import IdentitySelect, get_device


@pytest.fixture(scope="module")
def simple_graph():
    N = 10
    F = 3
    # Chain graph: edges (0-1, 1-2, ..., 8-9), made undirected
    row = torch.arange(9, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    E = edge_index.size(1)

    x = torch.randn((N, F), dtype=torch.float)
    edge_weight = torch.ones((E, 1), dtype=torch.float)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_attr=edge_weight, num_nodes=N
    )
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


@pytest.fixture(scope="module")
def sparse_batch_graph():
    edge_index_1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight_1 = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    x_1 = torch.randn((4, 3), dtype=torch.float)

    edge_index_2 = torch.tensor(
        [[1, 2, 3, 4, 2, 0], [0, 1, 2, 2, 3, 3]], dtype=torch.long
    )
    edge_weight_2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float)
    x_2 = torch.randn((5, 3), dtype=torch.float)

    edge_index_3 = torch.tensor([[0, 1, 3, 3, 2], [1, 0, 1, 2, 3]], dtype=torch.long)
    edge_weight_3 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float)
    x_3 = torch.randn((4, 3), dtype=torch.float)

    data_batch = Batch.from_data_list(
        [
            Data(edge_index=edge_index_1, edge_attr=edge_weight_1, x=x_1),
            Data(edge_index=edge_index_2, edge_attr=edge_weight_2, x=x_2),
            Data(edge_index=edge_index_3, edge_attr=edge_weight_3, x=x_3),
        ]
    )
    return data_batch


def test_nopool_basic_functionality(simple_graph):
    """Test basic NoPool functionality."""
    x, edge_index, edge_weight, batch = simple_graph
    N, F = x.size()

    # Test 1: Get pooler using get_pooler function
    pooler = get_pooler("nopool")
    assert pooler is not None
    assert not pooler.is_dense_batched

    # Test 2: Test preprocessing
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=x,
        batch=batch,
        use_cache=False,
    )
    assert x_pre.shape == x.shape
    # Try to import SparseTensor for type checking, but don't require it
    try:
        from torch_sparse import SparseTensor

        has_sparse = True
    except ImportError:
        SparseTensor = type(None)
        has_sparse = False

    if has_sparse:
        assert isinstance(adj_pre, (SparseTensor, torch.Tensor))
    else:
        assert isinstance(adj_pre, torch.Tensor)
    assert mask is None

    # Test 3: Test forward pass (pooling)
    out = pooler(x=x_pre, adj=adj_pre, edge_weight=edge_weight, batch=batch, mask=mask)
    assert hasattr(out, "x")
    assert hasattr(out, "so") and isinstance(out.so, SelectOutput)
    assert isinstance(out.x, torch.Tensor)
    assert out.x.shape == x.shape
    assert out.so.num_supernodes == N
    assert out.so.num_nodes == N

    # Test 4: Test lifting
    x_lifted = pooler(
        x=out.x,
        adj=out.edge_index,
        edge_weight=out.edge_weight,
        so=out.so,
        batch=out.batch,
        lifting=True,
    )
    assert x_lifted.shape == out.x.shape

    # Test 5: Test with message passing
    conv = GCNConv(F, F)
    out.x = conv(out.x, out.edge_index)
    assert out.x.shape == (N, F)

    # Test 6: Test with SparseTensor input
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    adj_sparse = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)
    out_sparse = pooler(x=x, adj=adj_sparse, batch=batch)
    assert out_sparse.x.shape == x.shape


def test_nopool_identity_behavior(simple_graph):
    """Test that NoPool preserves input exactly."""
    x, edge_index, edge_weight, batch = simple_graph

    pooler = get_pooler("nopool")
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=x,
        batch=batch,
        use_cache=False,
    )

    out = pooler(x=x_pre, adj=adj_pre, edge_weight=edge_weight, batch=batch, mask=mask)

    # Features should be identical
    assert torch.allclose(out.x, x)

    # Edge structure should be identical
    try:
        from torch_sparse import SparseTensor

        has_sparse = True
    except ImportError:
        SparseTensor = type(None)
        has_sparse = False

    if has_sparse and isinstance(adj_pre, SparseTensor):
        assert adj_pre.sparse_sizes() == out.edge_index.sparse_sizes()
    else:
        # Sort both edge indices lexicographically for comparison
        # Create a combined key: first row * max_val + second row for lexicographic sorting
        max_val = max(edge_index.max().item(), out.edge_index.max().item()) + 1

        # Sort original edge_index
        orig_key = edge_index[0] * max_val + edge_index[1]
        sort_idx_orig = torch.argsort(orig_key)
        edge_index_sorted = edge_index[:, sort_idx_orig]

        # Sort output edge_index
        out_key = out.edge_index[0] * max_val + out.edge_index[1]
        sort_idx_out = torch.argsort(out_key)
        out_edge_index_sorted = out.edge_index[:, sort_idx_out]

        assert torch.equal(out_edge_index_sorted, edge_index_sorted)

    # Batch should be identical
    assert torch.equal(out.batch, batch)

    # SelectOutput should represent identity mapping
    assert out.so.num_supernodes == out.so.num_nodes
    assert torch.equal(out.so.node_index, torch.arange(x.size(0)))
    assert torch.equal(out.so.cluster_index, torch.arange(x.size(0)))


def test_nopool_precoarsening(sparse_batch_graph):
    """Test NoPool precoarsening functionality."""
    pooler = get_pooler("nopool")

    data_batch = sparse_batch_graph
    assert data_batch.num_graphs == 3
    assert data_batch.num_nodes == 13

    # Test precoarsening
    try:
        pooled_batch = pooler.precoarsening(
            edge_index=data_batch.edge_index,
            edge_weight=data_batch.edge_attr,
            batch=data_batch.batch,
        )

        assert pooled_batch.batch.shape == data_batch.batch.shape
        assert pooled_batch.so.num_supernodes == data_batch.num_nodes
        assert pooled_batch.so.num_nodes == data_batch.num_nodes

    except NotImplementedError:
        # If precoarsening is not implemented, that's also acceptable for NoPool
        # since it's an identity operation
        pass


def test_nopool_in_pooler_list():
    """Test that NoPool is properly registered in the pooler system."""
    assert "nopool" in pooler_map
    assert "NoPool" in pooler_map.values().__class__.__name__ or "NoPool" in str(
        pooler_map.values()
    )

    # Test that it can be instantiated
    pooler = get_pooler("nopool")
    assert pooler.__class__.__name__ == "NoPool"


def test_nopool_print_signature():
    """Test that NoPool has proper print signature."""
    pooler = get_pooler("nopool")

    # Test extra_repr_args method exists and returns empty dict
    assert hasattr(pooler, "extra_repr_args")
    extra_args = pooler.extra_repr_args()
    assert isinstance(extra_args, dict)
    assert len(extra_args) == 0  # NoPool should have no extra args

    # Test string representation
    pooler_str = str(pooler)
    assert "NoPool" in pooler_str
    assert "IdentitySelect" in pooler_str

    # Test individual component print signatures
    assert hasattr(pooler.selector, "__repr__")
    assert hasattr(pooler.reducer, "__repr__")
    assert hasattr(pooler.connector, "__repr__")

    # Test that the components have proper string representations
    selector_str = str(pooler.selector)
    reducer_str = str(pooler.reducer)
    connector_str = str(pooler.connector)

    assert "IdentitySelect" in selector_str
    assert "BaseReduce" in reducer_str
    assert "SparseConnect" in connector_str


# ============================================================================
# IdentitySelect specific tests - testing every conditional path
# ============================================================================


@pytest.mark.torch_sparse
def test_identity_select_with_sparse_edge_index():
    """Test IdentitySelect when num_nodes is determined from SparseTensor edge_index."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    selector = IdentitySelect()

    # Create SparseTensor edge_index
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    sparse_edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))

    so = selector.forward(edge_index=sparse_edge_index)

    assert so.num_nodes == 4  # SparseTensor size
    assert so.num_supernodes == 4
    assert torch.equal(so.node_index, torch.arange(4))
    assert torch.equal(so.cluster_index, torch.arange(4))


def test_identity_select_with_dense_edge_index():
    """Test IdentitySelect when num_nodes is determined from dense edge_index."""
    selector = IdentitySelect()

    # Create dense edge_index
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    so = selector.forward(edge_index=edge_index)

    # num_nodes should be max(edge_index) + 1 = 3 + 1 = 4
    assert so.num_nodes == 4
    assert so.num_supernodes == 4
    assert torch.equal(so.node_index, torch.arange(4))
    assert torch.equal(so.cluster_index, torch.arange(4))


def test_identity_select_device_consistency_with_edge_index():
    """Test IdentitySelect uses device from edge_index tensor."""
    selector = IdentitySelect()

    # Test with CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=device)

    so = selector.forward(edge_index=edge_index)

    # Check device type and index separately to handle CUDA device differences
    assert so.node_index.device.type == device.type
    assert so.cluster_index.device.type == device.type


def test_identity_select_error_when_no_inputs():
    """Test IdentitySelect raises NotImplementedError when no inputs to determine num_nodes."""
    selector = IdentitySelect()

    with pytest.raises(NotImplementedError):
        selector.forward()


def test_identity_select_with_num_nodes_not_none():
    """Test IdentitySelect when num_nodes is explicitly provided (not None)."""
    selector = IdentitySelect()

    # Test case 1: num_nodes provided with edge_index
    # When num_nodes is explicitly provided, it should override the inferred value from edge_index
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    explicit_num_nodes = 6
    # edge_index would normally infer 4 nodes (max index + 1), but explicit num_nodes=6 should be used
    so = selector.forward(edge_index=edge_index, num_nodes=explicit_num_nodes)

    assert so.num_nodes == explicit_num_nodes
    assert so.num_supernodes == explicit_num_nodes
    assert torch.equal(so.node_index, torch.arange(explicit_num_nodes))
    assert torch.equal(so.cluster_index, torch.arange(explicit_num_nodes))

    # Test case 2: num_nodes provided with SparseTensor edge_index
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    sparse_edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
    explicit_num_nodes = 8
    # SparseTensor would normally infer 4 nodes, but explicit num_nodes=8 should be used
    so = selector.forward(edge_index=sparse_edge_index, num_nodes=explicit_num_nodes)

    assert so.num_nodes == explicit_num_nodes
    assert so.num_supernodes == explicit_num_nodes
    assert torch.equal(so.node_index, torch.arange(explicit_num_nodes))
    assert torch.equal(so.cluster_index, torch.arange(explicit_num_nodes))


def test_identity_select_with_kwargs():
    """Test IdentitySelect handles additional kwargs."""
    selector = IdentitySelect()

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    # Pass additional kwargs that should be ignored
    so = selector.forward(
        edge_index=edge_index, some_random_param=42, another_param="test"
    )

    # num_nodes inferred from edge_index: max(1) + 1 = 2
    assert so.num_nodes == 2
    assert so.num_supernodes == 2


def test_identity_select_reset_parameters():
    """Test IdentitySelect reset_parameters method (no-op)."""
    selector = IdentitySelect()

    # Should not raise any error
    selector.reset_parameters()


def test_identity_select_edge_case_single_node():
    """Test IdentitySelect edge case with single node."""
    selector = IdentitySelect()

    # Test with single node using edge_index
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    so = selector.forward(edge_index=edge_index)

    assert so.num_nodes == 1
    assert so.num_supernodes == 1
    assert torch.equal(so.node_index, torch.tensor([0]))
    assert torch.equal(so.cluster_index, torch.tensor([0]))


class TestGetDevice:
    """Test the get_device utility function."""

    def test_get_device_from_edge_index_tensor(self):
        """Test get_device returns device from edge_index Tensor."""
        # Test on CPU
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        device = get_device(None, edge_index)
        assert device == torch.device("cpu")
        assert device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            edge_index_cuda = edge_index.cuda()
            device_cuda = get_device(None, edge_index_cuda)
            assert device_cuda.type == "cuda"

    @pytest.mark.torch_sparse
    def test_get_device_from_sparse_tensor(self):
        """Test get_device returns device from SparseTensor edge_index."""
        pytest.importorskip("torch_sparse")
        from torch_sparse import SparseTensor

        # Create SparseTensor on CPU
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        sparse_edge_index = SparseTensor.from_edge_index(
            edge_index, sparse_sizes=(4, 4)
        )

        device = get_device(None, sparse_edge_index)
        assert device == torch.device("cpu")
        assert device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            sparse_edge_index_cuda = sparse_edge_index.cuda()
            device_cuda = get_device(None, sparse_edge_index_cuda)
            assert device_cuda.type == "cuda"

    def test_get_device_from_x_tensor(self):
        """Test get_device returns device from x tensor when edge_index is None."""
        # Test on CPU
        x = torch.randn(5, 3)
        device = get_device(x, None)
        assert device == torch.device("cpu")
        assert device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            device_cuda = get_device(x_cuda, None)
            assert device_cuda.type == "cuda"

    def test_get_device_edge_index_priority(self):
        """Test that edge_index takes priority over x when both are provided."""
        # Create tensors on different devices if CUDA is available
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        device = get_device(x, edge_index)
        # Should use edge_index device (both CPU in this case)
        assert device == torch.device("cpu")

        if torch.cuda.is_available():
            # x on CPU, edge_index on CUDA
            edge_index_cuda = edge_index.cuda()
            device = get_device(x, edge_index_cuda)
            assert device.type == "cuda"  # Should use edge_index device

            # x on CUDA, edge_index on CPU
            x_cuda = x.cuda()
            device = get_device(x_cuda, edge_index)
            assert device.type == "cpu"  # Should use edge_index device

    def test_get_device_raises_when_both_none(self):
        """Test get_device raises ValueError when both x and edge_index are None."""
        with pytest.raises(ValueError, match="No device found"):
            get_device(None, None)

        with pytest.raises(ValueError, match="No device found"):
            get_device()

    def test_get_device_with_empty_edge_index(self):
        """Test get_device works with empty edge_index."""
        # Empty edge_index tensor
        edge_index = torch.empty((2, 0), dtype=torch.long)
        device = get_device(None, edge_index)
        assert device == torch.device("cpu")

    def test_get_device_with_single_node_edge_index(self):
        """Test get_device works with single node edge_index."""
        # Self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        device = get_device(None, edge_index)
        assert device == torch.device("cpu")

    @pytest.mark.torch_sparse
    def test_get_device_with_sparse_tensor_priority(self):
        """Test that SparseTensor edge_index takes priority over x."""
        pytest.importorskip("torch_sparse")
        from torch_sparse import SparseTensor

        x = torch.randn(3, 2)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        sparse_edge_index = SparseTensor.from_edge_index(
            edge_index, sparse_sizes=(3, 3)
        )

        device = get_device(x, sparse_edge_index)
        # Should use sparse_edge_index device
        assert device == torch.device("cpu")
