import pytest
import torch
from scipy.sparse import csr_matrix
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import add_self_loops, to_undirected
from torch_sparse import SparseTensor

from tgp.connect import (
    Connect,
    DenseConnectUnbatched,
    KronConnect,
    SparseConnect,
    sparse_connect,
)
from tgp.reduce import BaseReduce
from tgp.select import SelectOutput, TopkSelect
from tgp.select.kmis_select import KMISSelect
from tgp.src import SRCPooling
from tgp.utils import connectivity_to_sparse_tensor


def test_connect_repr():
    # Test the __repr__ method of the Connect class
    connect_instance = Connect()
    repr_str = repr(connect_instance)
    assert isinstance(repr_str, str)


def test_sparse_connect_raises_runtime_error():
    """Calling sparse_connect with neither node_index nor cluster_index
    should hit the `else: raise RuntimeError` branch.
    """
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)

    # Neither node_index nor cluster_index provided:
    with pytest.raises(RuntimeError):
        _ = sparse_connect(
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_index=None,
            cluster_index=None,
        )

    # Check if the same error is raised for SparseConnect when s is dense and no node_index or cluster_index is provided
    s = torch.randn((2, 1), dtype=torch.float)
    so = SelectOutput(s=s)
    connector = SparseConnect()
    with pytest.raises(RuntimeError):
        _ = connector(
            edge_index=edge_index,
            edge_weight=edge_weight,
            so=so,
        )


def test_denseconn_unbatched_with_batch():
    """Test DenseConnectUnbatched with batch and remove_self_loops=False and degree_norm=False."""
    K = 4
    BS = 3
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    a = torch.tensor([1, 0, 0, 0], dtype=torch.float)
    b = torch.tensor([0, 1, 0, 0], dtype=torch.float)
    c = torch.tensor([0, 0, 1, 0], dtype=torch.float)
    d = torch.tensor([0, 0, 0, 1], dtype=torch.float)

    s1 = torch.stack([a, a, a], dim=0)
    s2 = torch.stack([b, c, b, b, c, b], dim=0)
    s3 = torch.stack([d, d, d, d], dim=0)
    my_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    dense_s = torch.cat([s1, s2, s3], dim=0)
    so = SelectOutput(s=dense_s)

    e1 = to_undirected(
        torch.tensor(data=[[0, 1, 1, 0], [1, 0, 1, 2]], dtype=torch.long)
    )
    e2 = to_undirected(
        torch.tensor(
            data=[[0, 1, 2, 2, 3, 3, 4], [1, 2, 3, 4, 4, 5, 5]], dtype=torch.long
        )
    )
    e3 = to_undirected(
        torch.tensor(data=[[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long)
    )

    d1 = Data(edge_index=e1, edge_weight=torch.rand(e1.size(1), dtype=torch.float))
    d2 = Data(edge_index=e2, edge_weight=torch.rand(e2.size(1), dtype=torch.float))
    d3 = Data(edge_index=e3, edge_weight=torch.rand(e3.size(1), dtype=torch.float))

    b = Batch.from_data_list([d1, d2, d3])
    batch = b.batch
    assert torch.all(b.batch == my_batch)

    edge_index = b.edge_index
    edge_weight = b.edge_weight

    # check with SparseTensor as Input
    adj = connectivity_to_sparse_tensor(edge_index, edge_weight)
    adj_pool, _ = connector(edge_index=adj, edge_weight=None, batch=batch, so=so)

    # check if it is a SparseTensor
    assert isinstance(adj_pool, SparseTensor)
    # check if the size is correct
    assert adj_pool.size(0) == adj_pool.size(0) == K * BS
    # check if the sum is ok
    adj_pool = adj_pool.to_dense()
    assert torch.isclose(d1.edge_weight.sum(), adj_pool[0, 0].sum())
    assert torch.isclose(d2.edge_weight.sum(), adj_pool[5:7, 5:7].sum())
    assert torch.isclose(d3.edge_weight.sum(), adj_pool[-1, -1].sum())

    # check with edge_index as Input
    edge_index_pool, edge_weight_pool = connector(
        edge_index=edge_index, edge_weight=edge_weight, batch=batch, so=so
    )

    # check if it is a SparseTensor
    assert isinstance(edge_index_pool, Tensor)
    assert isinstance(edge_weight_pool, Tensor)
    # check the size is correct
    assert edge_index_pool.size(1) == 5
    # check if the sum is ok
    assert torch.isclose(d1.edge_weight.sum(), edge_weight_pool[0].sum())
    assert torch.isclose(d2.edge_weight.sum(), edge_weight_pool[1:-1].sum())
    assert torch.isclose(d3.edge_weight.sum(), edge_weight_pool[-1].sum())
    # check if the sum is ok


def test_denseconn_spt():
    """Test the behavior of DenseConnectUnbatched with remove_self_loops=False and degree_norm=False."""
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    s = torch.eye(3, dtype=torch.float)
    so = SelectOutput(s=s)
    edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.5, 0.7], dtype=torch.float)
    adj_pool, _ = connector(edge_index=edge_index, edge_weight=edge_weight, so=so)

    # check if the diagonal is zero
    assert edge_index.size(0) == adj_pool.size(0)


def test_denseconn_spt_invalid_edge_index_type():
    """Passing an unsupported edge_index type (e.g., a Python list) into DenseConnectUnbatched.forward
    should raise a ValueError.
    """
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    s = torch.eye(2).unsqueeze(0)
    so = SelectOutput(s=s)
    invalid_edge_index = [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="Edge index must be of type"):
        _ = connector(edge_index=invalid_edge_index, edge_weight=None, so=so)


def test_denseconn_spt_single_graph_dense_s_dim3_error():
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    s = torch.randn(2, 3, 2)
    so = SelectOutput(s=s)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(
        ValueError,
        match="DenseConnectUnbatched expects a 2D assignment matrix for a single graph.",
    ):
        _ = connector(edge_index=edge_index, edge_weight=None, so=so)


def test_denseconn_spt_single_graph_dense_s_dim3_squeezed():
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    s = torch.eye(2).unsqueeze(0)
    so = SelectOutput(s=s)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    edge_index_pool, edge_weight_pool = connector(
        edge_index=edge_index, edge_weight=None, so=so
    )

    assert isinstance(edge_index_pool, Tensor)
    assert edge_index_pool.size(1) == 2
    assert isinstance(edge_weight_pool, Tensor)
    assert edge_weight_pool.numel() == 2


def test_denseconn_spt_batched_sparse_s_is_unsupported():
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    row = torch.arange(4, dtype=torch.long)
    col = torch.zeros(4, dtype=torch.long)
    s = SparseTensor(row=row, col=col, value=torch.ones(4), sparse_sizes=(4, 3))
    so = SelectOutput(s=s)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(TypeError, match="DenseConnectUnbatched expects a dense Tensor"):
        _ = connector(edge_index=edge_index, edge_weight=None, batch=batch, so=so)


def test_denseconn_spt_batched_dense_s_dim_error():
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    batch = torch.tensor([0, 1], dtype=torch.long)
    s = torch.randn(2, 2, 2)
    so = SelectOutput(s=s)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(
        ValueError,
        match="DenseConnectUnbatched expects a 2D assignment matrix for batched graphs.",
    ):
        _ = connector(edge_index=edge_index, edge_weight=None, batch=batch, so=so)


def test_denseconn_spt_batched_empty_edges():
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    s = torch.eye(4, 2)
    so = SelectOutput(s=s)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_index_pool, edge_weight_pool = connector(
        edge_index=edge_index, edge_weight=None, batch=batch, so=so
    )
    assert isinstance(edge_index_pool, Tensor)
    assert edge_index_pool.size(1) == 0
    assert isinstance(edge_weight_pool, Tensor)
    assert edge_weight_pool.numel() == 0


def test_denseconn_spt_filters_near_zero_edges_and_flattens_weights():
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    s = torch.eye(2, dtype=torch.float)
    so = SelectOutput(s=s)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([[1e-12], [1.0]], dtype=torch.float)

    edge_index_pool, edge_weight_pool = connector(
        edge_index=edge_index, edge_weight=edge_weight, so=so
    )

    assert isinstance(edge_index_pool, Tensor)
    assert edge_weight_pool.dim() == 1
    assert edge_weight_pool.numel() == 1
    assert torch.all(edge_weight_pool.abs() > 1e-8)


def test_denseconn_spt_resizes_sparse_adj_for_isolated_nodes():
    connector = DenseConnectUnbatched(remove_self_loops=False, degree_norm=False)
    s = torch.eye(3, dtype=torch.float)
    so = SelectOutput(s=s)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(2, 2))

    adj_pool, _ = connector(edge_index=adj, edge_weight=None, so=so)
    assert isinstance(adj_pool, SparseTensor)
    assert adj_pool.sparse_sizes() == (3, 3)


def test_kron_conn_without_ndp():
    N = 10
    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 0),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
    ]
    row = torch.tensor(
        [u for u, v in edge_list] + [v for u, v in edge_list], dtype=torch.long
    )
    col = torch.tensor(
        [v for u, v in edge_list] + [u for u, v in edge_list], dtype=torch.long
    )
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    x = torch.randn((N, 3))
    batch = torch.zeros(N, dtype=torch.long)

    # Test KronConnect with SparseTensor
    edge_index_spt = SparseTensor.from_edge_index(
        edge_index, edge_attr=edge_weight, sparse_sizes=(N, N)
    )
    data = Data(x=x, edge_index=edge_index_spt, batch=batch)
    data.num_nodes = N
    pooler = SRCPooling(
        selector=TopkSelect(in_channels=3, ratio=4, s_inv_op="transpose"),
        reducer=BaseReduce(),
        connector=KronConnect(),
    )

    so = pooler.select(
        x=data.x,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        batch=data.batch,
    )
    x_pool, batch_pool = pooler.reduce(x=data.x, so=so, batch=data.batch)
    adj_pool = pooler.connect(
        edge_index=data.edge_index, so=so, edge_weight=data.edge_weight
    )
    assert x_pool.shape[-2] == 4
    assert batch_pool.size(0) == x_pool.size(-2)
    assert adj_pool is not None

    # Test also with edge index
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, batch=batch)
    data.num_nodes = N
    pooler = SRCPooling(
        selector=TopkSelect(in_channels=3, ratio=4, s_inv_op="transpose"),
        reducer=BaseReduce(),
        connector=KronConnect(),
    )
    so = pooler.select(
        x=data.x,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        batch=data.batch,
    )
    x_pool, batch_pool = pooler.reduce(x=data.x, so=so, batch=data.batch)
    adj_pool = pooler.connect(
        edge_index=data.edge_index, so=so, edge_weight=data.edge_weight
    )
    assert x_pool.shape[-2] == 4
    assert batch_pool.size(0) == x_pool.size(-2)
    assert adj_pool is not None


def test_kronconnect_handles_singular_L():
    """Construct L so that the “complement” submatrix L_comp is singular,
    forcing the except-branch (Marquardt-Levenberg dampening) in Kron reduction.
    """
    data = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    L = csr_matrix(data)
    node_index = torch.tensor([0, 1], dtype=torch.long)
    num_nodes = 3
    cluster_index = torch.tensor([0, 1], dtype=torch.long)
    so = SelectOutput(
        s=None,
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=num_nodes,
        L=L,
    )
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    kc = KronConnect(sparse_threshold=0.0)
    adj_pool, _ = kc(edge_index=edge_index, so=so, edge_weight=edge_weight)
    assert isinstance(adj_pool, torch.Tensor)


def test_kronconnect_single_node_selection():
    """Test KronConnect when only 0 or 1 nodes are selected (line 88 coverage).
    This tests the early return path: Lnew = sp.csc_matrix(-np.ones((1, 1)))
    with both SparseTensor and regular tensor edge_index inputs.
    """
    # Create a simple 3-node graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    num_nodes = 3

    # Create a Laplacian matrix for the 3-node graph to avoid the conversion path
    L_data = [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]
    L = csr_matrix(L_data)

    # Test case 1: Select exactly 1 node with regular tensor edge_index and provided Laplacian
    node_index = torch.tensor([0], dtype=torch.long)  # Select only node 0
    cluster_index = torch.tensor([0], dtype=torch.long)
    so_with_L = SelectOutput(
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=num_nodes,
        num_supernodes=1,
        L=L,  # Provide Laplacian to avoid conversion
    )

    kc = KronConnect(sparse_threshold=0.0)
    adj_pool, edge_weight_pool = kc(
        edge_index=edge_index, so=so_with_L, edge_weight=edge_weight
    )

    # Should return a tensor (not SparseTensor) with 1x1 adjacency
    assert isinstance(adj_pool, torch.Tensor)
    assert isinstance(edge_weight_pool, torch.Tensor)

    # Test case 2: Select exactly 1 node with SparseTensor edge_index and provided Laplacian
    edge_index_spt = SparseTensor.from_edge_index(
        edge_index, edge_attr=edge_weight, sparse_sizes=(num_nodes, num_nodes)
    )

    adj_pool_spt, edge_weight_pool_spt = kc(
        edge_index=edge_index_spt, so=so_with_L, edge_weight=edge_weight
    )

    # Should return a SparseTensor with None edge weights
    assert isinstance(adj_pool_spt, SparseTensor)
    assert edge_weight_pool_spt is None

    # Test case 3: Select exactly 1 node WITHOUT provided Laplacian (tests conversion path)
    so_without_L = SelectOutput(
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=num_nodes,
        num_supernodes=1,
    )

    # With regular tensor edge_index (no conversion needed)
    adj_pool_no_L, edge_weight_pool_no_L = kc(
        edge_index=edge_index, so=so_without_L, edge_weight=edge_weight
    )
    assert isinstance(adj_pool_no_L, torch.Tensor)
    assert isinstance(edge_weight_pool_no_L, torch.Tensor)

    # With SparseTensor edge_index (should preserve SparseTensor output format)
    adj_pool_spt_no_L, edge_weight_pool_spt_no_L = kc(
        edge_index=edge_index_spt, so=so_without_L, edge_weight=edge_weight
    )
    # Should return a SparseTensor with None edge weights (preserving input format)
    assert isinstance(adj_pool_spt_no_L, SparseTensor)
    assert edge_weight_pool_spt_no_L is None


def test_sparse_connect_degree_norm_without_edge_weight():
    """Test sparse_connect with degree_norm=True and edge_weight=None.

    This tests the specific case where edge_weight is None but degree_norm is True,
    which should trigger the creation of ones tensor for edge weights.
    """
    # Create a simple graph with 4 nodes
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    num_nodes = 4
    num_supernodes = 2

    # Create cluster_index for coarsening (2 nodes per supernode)
    cluster_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Test with degree_norm=True and edge_weight=None
    adj_pool, edge_weight_pool = sparse_connect(
        edge_index=edge_index,
        edge_weight=None,
        cluster_index=cluster_index,
        num_nodes=num_nodes,
        num_supernodes=num_supernodes,
        degree_norm=True,
        remove_self_loops=True,
    )

    # Verify that edge weights were created and are not None
    assert edge_weight_pool is not None
    assert isinstance(edge_weight_pool, torch.Tensor)
    assert edge_weight_pool.size(0) == adj_pool.size(1)  # Same number of edges

    # Verify that the edge weights are normalized (should be between 0 and 1)
    assert torch.all(edge_weight_pool >= 0.0)
    assert torch.all(edge_weight_pool <= 1.0)

    # Test with SparseConnect class as well
    connector = SparseConnect(degree_norm=True)
    so = SelectOutput(
        cluster_index=cluster_index,
        num_nodes=num_nodes,
        num_supernodes=num_supernodes,
    )

    adj_pool_class, edge_weight_pool_class = connector(
        edge_index=edge_index,
        edge_weight=None,
        so=so,
    )

    # Verify that edge weights were created and are not None
    assert edge_weight_pool_class is not None
    assert isinstance(edge_weight_pool_class, torch.Tensor)
    assert edge_weight_pool_class.size(0) == adj_pool_class.size(1)

    # Verify that the edge weights are normalized
    assert torch.all(edge_weight_pool_class >= 0.0)
    assert torch.all(edge_weight_pool_class <= 1.0)


def test_kmis_kron_connector():
    """Test KronConnect with KMIS pooling.

    This tests the code path in KronConnect where KMIS is used (lines 93-100),
    specifically the validation that MIS indices are within bounds.
    """
    # Create a simple graph
    N = 10
    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (0, 9),
    ]
    row = torch.tensor(
        [u for u, v in edge_list] + [v for u, v in edge_list], dtype=torch.long
    )
    col = torch.tensor(
        [v for u, v in edge_list] + [u for u, v in edge_list], dtype=torch.long
    )
    edge_index = torch.stack([row, col], dim=0)
    x = torch.randn((N, 3))
    batch = torch.zeros(N, dtype=torch.long)

    # Create KMIS selector
    kmis_selector = KMISSelect(in_channels=3, order_k=2, scorer="degree")

    # Select nodes using KMIS
    so = kmis_selector(x=x, edge_index=edge_index, batch=batch)

    # Verify SelectOutput has the expected attributes
    assert hasattr(so, "mis"), "SelectOutput should have 'mis' attribute from KMIS"
    assert hasattr(so, "num_nodes")
    assert hasattr(so, "num_supernodes")

    # Verify MIS indices are within bounds (this is what line 93-100 checks)
    assert torch.all(so.mis < so.num_nodes), "MIS indices should be within num_nodes"
    assert torch.all(so.mis >= 0), "MIS indices should be non-negative"

    # Test KronConnect with KMIS SelectOutput
    kron_connector = KronConnect()
    edge_index_pool, edge_weight_pool = kron_connector(
        edge_index=edge_index,
        so=so,
        edge_weight=None,
        batch_pooled=batch[: so.num_supernodes],
    )

    # Verify output is valid
    assert isinstance(edge_index_pool, torch.Tensor)
    assert edge_index_pool.size(0) == 2  # Should have 2 rows (source, target)

    # Verify pooled edge indices are within bounds
    if edge_index_pool.size(1) > 0:
        assert torch.all(edge_index_pool < so.num_supernodes)
        assert torch.all(edge_index_pool >= 0)


def test_kmis_kron_invalid_mis_indices():
    """Test that KronConnect raises ValueError when MIS indices are out of bounds.

    This directly tests the validation on lines 93-100 in kron_conn.py.
    """
    # Create a graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3

    # Create a MALFORMED SelectOutput with out-of-bounds MIS indices
    # This should never happen in practice, but tests the defensive check
    invalid_mis = torch.tensor([0, 5], dtype=torch.long)  # 5 is out of bounds!

    so = SelectOutput(
        num_nodes=num_nodes,
        num_supernodes=2,
        node_index=torch.arange(num_nodes),
        cluster_index=torch.tensor([0, 0, 1]),
        mis=invalid_mis,  # Invalid: 5 >= num_nodes (3)
    )

    kron_connector = KronConnect()

    # This should raise ValueError due to out-of-bounds MIS indices
    with pytest.raises(ValueError, match="MIS indices out of range"):
        _ = kron_connector(
            edge_index=edge_index,
            so=so,
            edge_weight=None,
            batch_pooled=torch.zeros(2, dtype=torch.long),
        )


if __name__ == "__main__":
    pytest.main([__file__])
