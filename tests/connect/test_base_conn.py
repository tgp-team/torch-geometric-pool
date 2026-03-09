"""Tests for tgp.connect.base_conn module.

Tests the Connect base class, sparse_connect function, and SparseConnect class.
"""

import pytest
import torch

from tgp.connect import Connect, SparseConnect, sparse_connect
from tgp.select import SelectOutput


class TestConnect:
    """Test the Connect base class."""

    def test_connect_repr(self):
        """Test the __repr__ method of the Connect class."""
        connect_instance = Connect()
        repr_str = repr(connect_instance)
        assert isinstance(repr_str, str)
        assert "Connect()" in repr_str

    def test_connect_forward_not_implemented(self):
        """Test that Connect.forward raises NotImplementedError."""
        connect_instance = Connect()
        so = SelectOutput(s=torch.eye(3))
        with pytest.raises(NotImplementedError):
            connect_instance.forward(
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                so=so,
            )


class TestSparseConnect:
    """Test the SparseConnect class."""

    def test_sparse_connect_raises_runtime_error(self):
        """Test that sparse_connect raises RuntimeError when neither node_index nor cluster_index is provided."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)

        # Neither node_index nor cluster_index provided
        with pytest.raises(RuntimeError):
            sparse_connect(
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
            connector(
                edge_index=edge_index,
                edge_weight=edge_weight,
                so=so,
            )

    def test_sparse_connect_with_node_index(self, pooler_test_graph_sparse):
        """Test sparse_connect with node_index (e.g., TopK pooling)."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create node_index for top-k selection (select first half of nodes)
        k = num_nodes // 2
        node_index = torch.arange(k, dtype=torch.long)

        # Test sparse_connect function
        adj_pool, edge_weight_pool = sparse_connect(
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_index=node_index,
            num_nodes=num_nodes,
            num_supernodes=k,
            remove_self_loops=True,
        )

        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.size(0) == 2  # edge_index format
        if edge_weight_pool is not None:
            assert edge_weight_pool.size(0) == adj_pool.size(1)

        # Test SparseConnect class
        cluster_index = torch.arange(k, dtype=torch.long)
        so = SelectOutput(
            node_index=node_index,
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=k,
        )
        connector = SparseConnect(remove_self_loops=True)
        adj_pool_class, edge_weight_pool_class = connector(
            edge_index=edge_index,
            edge_weight=edge_weight,
            so=so,
        )

        assert isinstance(adj_pool_class, torch.Tensor)
        assert adj_pool_class.size(0) == 2

    def test_sparse_connect_with_cluster_index(self, pooler_test_graph_sparse):
        """Test sparse_connect with cluster_index (e.g., Graclus, NDP)."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)
        num_supernodes = num_nodes // 2

        # Create cluster_index (2 nodes per cluster)
        cluster_index = torch.arange(num_nodes, dtype=torch.long) // 2

        # Test sparse_connect function
        adj_pool, edge_weight_pool = sparse_connect(
            edge_index=edge_index,
            edge_weight=edge_weight,
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
            remove_self_loops=True,
            reduce_op="sum",
        )

        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.size(0) == 2
        if edge_weight_pool is not None:
            assert edge_weight_pool.size(0) == adj_pool.size(1)

        # Test SparseConnect class
        so = SelectOutput(
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
        )
        connector = SparseConnect(remove_self_loops=True, reduce_op="sum")
        adj_pool_class, edge_weight_pool_class = connector(
            edge_index=edge_index,
            edge_weight=edge_weight,
            so=so,
        )

        assert isinstance(adj_pool_class, torch.Tensor)
        assert adj_pool_class.size(0) == 2

    def test_sparse_connect_degree_norm_without_edge_weight(self):
        """Test sparse_connect with degree_norm=True and edge_weight=None."""
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
        assert edge_weight_pool.size(0) == adj_pool.size(1)

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

    @pytest.mark.torch_sparse
    def test_sparse_connect_with_torch_coo(self):
        """Test sparse_connect with torch COO sparse tensor input."""
        pytest.importorskip("torch_sparse")

        # Create a simple graph with 4 nodes
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
        num_nodes = 4
        num_supernodes = 2

        # Convert to torch COO sparse tensor
        edge_index_coo = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(num_nodes, num_nodes)
        ).coalesce()

        # Create cluster_index for coarsening (2 nodes per supernode)
        cluster_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Test with torch COO sparse tensor input
        adj_pool, edge_weight_pool = sparse_connect(
            edge_index=edge_index_coo,
            edge_weight=None,  # Edge weights are embedded in the sparse tensor
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
            degree_norm=False,
            remove_self_loops=True,
        )

        # Verify that output is a torch COO sparse tensor
        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.is_sparse
        assert edge_weight_pool is None  # Should be None when output is torch COO

        # Verify the shape of the pooled adjacency
        assert adj_pool.shape == (num_supernodes, num_supernodes)

        # Test with SparseConnect class as well
        connector = SparseConnect()
        so = SelectOutput(
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
        )

        adj_pool_class, edge_weight_pool_class = connector(
            edge_index=edge_index_coo,
            edge_weight=None,
            so=so,
        )

        # Verify that output is a torch COO sparse tensor
        assert isinstance(adj_pool_class, torch.Tensor)
        assert adj_pool_class.is_sparse
        assert edge_weight_pool_class is None  # Should be None when output is torch COO

        # Verify the shape of the pooled adjacency
        assert adj_pool_class.shape == (num_supernodes, num_supernodes)

    def test_sparse_connect_repr(self):
        """Test SparseConnect __repr__ method."""
        connector = SparseConnect(
            reduce_op="mean",
            remove_self_loops=False,
            edge_weight_norm=True,
            degree_norm=True,
        )
        repr_str = repr(connector)
        assert "SparseConnect" in repr_str
        assert "reduce_op=mean" in repr_str
        assert "remove_self_loops=False" in repr_str
        assert "edge_weight_norm=True" in repr_str
        assert "degree_norm=True" in repr_str

    def test_sparse_connect_edge_weight_norm_requires_batch_pooled(self):
        """Test that SparseConnect raises AssertionError when edge_weight_norm=True but batch_pooled=None."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
        node_index = torch.tensor([0, 1], dtype=torch.long)
        so = SelectOutput(
            node_index=node_index,
            cluster_index=torch.arange(node_index.numel(), dtype=torch.long),
            num_nodes=2,
            num_supernodes=2,
        )

        connector = SparseConnect(edge_weight_norm=True)
        with pytest.raises(AssertionError, match="batch_pooled parameter is required"):
            connector(
                edge_index=edge_index,
                edge_weight=edge_weight,
                so=so,
                batch_pooled=None,
            )
