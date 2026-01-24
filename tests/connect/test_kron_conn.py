"""Tests for tgp.connect.kron_conn module.

Tests the KronConnect class for Kron reduction-based pooling.
"""

import pytest
import torch
from scipy.sparse import csr_matrix

from tgp.connect import KronConnect
from tgp.select import SelectOutput
from tgp.select.kmis_select import KMISSelect


class TestKronConnect:
    """Test the KronConnect class."""

    def test_kron_connect_repr(self):
        """Test KronConnect __repr__ method."""
        connector = KronConnect(sparse_threshold=0.01)
        repr_str = repr(connector)
        assert "KronConnect" in repr_str
        assert "sparse_threshold=0.01" in repr_str

    @pytest.mark.torch_sparse
    def test_kron_connect_without_ndp(self, pooler_test_graph_sparse):
        """Test KronConnect without NDP (Laplacian not provided)."""
        pytest.importorskip("torch_sparse")
        from torch_sparse import SparseTensor

        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create SelectOutput without Laplacian (will trigger warning and conversion)
        k = num_nodes // 2
        node_index = torch.arange(k, dtype=torch.long)
        so = SelectOutput(
            node_index=node_index,
            num_nodes=num_nodes,
            num_supernodes=k,
        )

        # Test with SparseTensor
        edge_index_spt = SparseTensor.from_edge_index(
            edge_index, edge_attr=edge_weight, sparse_sizes=(num_nodes, num_nodes)
        )

        connector = KronConnect()
        adj_pool_spt, edge_weight_pool_spt = connector(
            edge_index=edge_index_spt, so=so, edge_weight=edge_weight
        )

        assert adj_pool_spt is not None
        assert isinstance(adj_pool_spt, SparseTensor)
        assert edge_weight_pool_spt is None

        # Test also with edge index
        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index, so=so, edge_weight=edge_weight
        )

        assert adj_pool is not None
        assert isinstance(adj_pool, torch.Tensor)
        assert edge_weight_pool is not None

    def test_kron_connect_with_torch_coo(self, pooler_test_graph_sparse):
        """Test KronConnect with torch COO sparse tensor input."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Convert to torch COO sparse tensor
        edge_index_coo = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(num_nodes, num_nodes)
        ).coalesce()

        # Create SelectOutput
        k = num_nodes // 2
        node_index = torch.arange(k, dtype=torch.long)
        so = SelectOutput(
            node_index=node_index,
            num_nodes=num_nodes,
            num_supernodes=k,
        )

        connector = KronConnect()
        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index_coo, so=so, edge_weight=None
        )

        # Verify output is torch COO sparse tensor
        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.is_sparse
        assert edge_weight_pool is None  # Edge weights are embedded in sparse tensor

    def test_kron_connect_handles_singular_L(self):
        """Test KronConnect handles singular Laplacian complement matrix."""
        # Construct L so that the "complement" submatrix L_comp is singular,
        # forcing the except-branch (Marquardt-Levenberg dampening) in Kron reduction.
        data = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
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

    @pytest.mark.torch_sparse
    def test_kron_connect_single_node_selection(self):
        """Test KronConnect when only 0 or 1 nodes are selected."""
        pytest.importorskip("torch_sparse")
        from torch_sparse import SparseTensor

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

    def test_kron_connect_with_kmis(self, pooler_test_graph_sparse):
        """Test KronConnect with KMIS pooling."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse

        # Create KMIS selector
        kmis_selector = KMISSelect(in_channels=x.size(1), order_k=2, scorer="degree")

        # Select nodes using KMIS
        so = kmis_selector(x=x, edge_index=edge_index, batch=batch)

        # Verify SelectOutput has the expected attributes
        assert hasattr(so, "mis"), "SelectOutput should have 'mis' attribute from KMIS"
        assert hasattr(so, "num_nodes")
        assert hasattr(so, "num_supernodes")

        # Verify MIS indices are within bounds
        assert torch.all(so.mis < so.num_nodes), (
            "MIS indices should be within num_nodes"
        )
        assert torch.all(so.mis >= 0), "MIS indices should be non-negative"

        # Test KronConnect with KMIS SelectOutput
        kron_connector = KronConnect()
        edge_index_pool, edge_weight_pool = kron_connector(
            edge_index=edge_index,
            so=so,
            edge_weight=edge_weight,
        )

        # Verify output is valid
        assert isinstance(edge_index_pool, torch.Tensor)
        assert edge_index_pool.size(0) == 2  # Should have 2 rows (source, target)

        # Verify pooled edge indices are within bounds
        if edge_index_pool.size(1) > 0:
            assert torch.all(edge_index_pool < so.num_supernodes)
            assert torch.all(edge_index_pool >= 0)

    def test_kron_connect_invalid_mis_indices(self):
        """Test that KronConnect raises ValueError when MIS indices are out of bounds."""
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
            )

    def test_kron_connect_sparse_threshold(self, pooler_test_graph_sparse):
        """Test KronConnect sparse_threshold parameter."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create SelectOutput
        k = num_nodes // 2
        node_index = torch.arange(k, dtype=torch.long)
        so = SelectOutput(
            node_index=node_index,
            num_nodes=num_nodes,
            num_supernodes=k,
        )

        # Test with high threshold (should remove more edges)
        connector_high = KronConnect(sparse_threshold=10.0)
        adj_pool_high, _ = connector_high(
            edge_index=edge_index, so=so, edge_weight=edge_weight
        )

        # Test with low threshold (should keep more edges)
        connector_low = KronConnect(sparse_threshold=0.0)
        adj_pool_low, _ = connector_low(
            edge_index=edge_index, so=so, edge_weight=edge_weight
        )

        # Low threshold should have at least as many edges as high threshold
        if isinstance(adj_pool_low, torch.Tensor) and adj_pool_low.size(0) == 2:
            if isinstance(adj_pool_high, torch.Tensor) and adj_pool_high.size(0) == 2:
                assert adj_pool_low.size(1) >= adj_pool_high.size(1)
