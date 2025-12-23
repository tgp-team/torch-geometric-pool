#!/usr/bin/env python3
"""Test edge weight normalization across all connect methods.
Tests: SparseConnect, DenseConnect, DenseConnectSPT
"""

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor

from tgp.connect import DenseConnect, DenseConnectSPT, SparseConnect
from tgp.reduce import BaseReduce
from tgp.select import TopkSelect


class TestEdgeWeightNormalization:
    """Test edge weight normalization functionality."""

    def create_batch_graphs(self):
        """Create batched test graphs - the proper way to test."""
        # Graph 1: 4 nodes
        data1 = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            edge_weight=torch.tensor([1.0, 2.0, 4.0, 3.0, 6.0, 2.0]),  # max = 6.0
        )

        # Graph 2: 3 nodes
        data2 = Data(
            x=torch.randn(3, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_weight=torch.tensor([8.0, 2.0, 4.0]),  # max = 8.0
        )

        return Batch.from_data_list([data1, data2])

    def create_single_graph_batch(self):
        """Create a single graph in batch format (batch_size=1)."""
        data = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            edge_weight=torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 3.0]),  # max = 10.0
        )
        return Batch.from_data_list([data])

    def get_select_output_and_batch_pooled(self, batch_data, ratio=0.5):
        """Get SelectOutput and compute batch_pooled."""
        selector = TopkSelect(in_channels=batch_data.x.size(1), ratio=ratio)
        so = selector(batch_data.x, batch=batch_data.batch)
        batch_pooled = BaseReduce.reduce_batch(so, batch_data.batch)
        return so, batch_pooled

    def create_dense_adjacency(self, batch_data):
        """Create dense adjacency matrices for DenseConnect testing."""
        batch_size = batch_data.batch.max().item() + 1
        max_nodes = (
            (batch_data.batch == 0).sum().item()
        )  # Assume same size for simplicity

        # Create dense adjacency matrices
        adj_dense = torch.zeros(batch_size, max_nodes, max_nodes)

        for batch_idx in range(batch_size):
            # Get nodes for this graph
            node_mask = batch_data.batch == batch_idx
            graph_nodes = node_mask.sum().item()

            # Get edges for this graph
            edge_mask = (
                node_mask[batch_data.edge_index[0]]
                & node_mask[batch_data.edge_index[1]]
            )
            graph_edges = batch_data.edge_index[:, edge_mask]
            graph_weights = batch_data.edge_weight[edge_mask]

            # Convert to local node indices
            node_mapping = torch.zeros(batch_data.num_nodes, dtype=torch.long)
            node_mapping[node_mask] = torch.arange(graph_nodes)
            local_edges = node_mapping[graph_edges]

            # Fill dense matrix
            adj_dense[batch_idx, local_edges[0], local_edges[1]] = graph_weights

        return adj_dense

    # ===== SPARSE CONNECT TESTS =====

    def test_sparse_connect_assertion_error(self):
        """Test that SparseConnect raises AssertionError when batch_pooled is None."""
        batch_data = self.create_batch_graphs()
        so, _ = self.get_select_output_and_batch_pooled(batch_data)

        connector = SparseConnect(edge_weight_norm=True)

        with pytest.raises(AssertionError, match="batch_pooled parameter is required"):
            connector(
                batch_data.edge_index,
                so,
                edge_weight=batch_data.edge_weight,
                batch_pooled=None,
            )

    def test_sparse_connect_with_batch_pooled(self):
        """Test SparseConnect with proper batch_pooled parameter."""
        batch_data = self.create_batch_graphs()
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # Test without normalization
        connector = SparseConnect(edge_weight_norm=False)
        edge_index_orig, edge_weight_orig = connector(
            batch_data.edge_index, so, edge_weight=batch_data.edge_weight
        )

        # Test with normalization
        connector_norm = SparseConnect(edge_weight_norm=True)
        edge_index_norm, edge_weight_norm = connector_norm(
            batch_data.edge_index,
            so,
            edge_weight=batch_data.edge_weight,
            batch_pooled=batch_pooled,
        )

        # Check that structure is preserved
        torch.testing.assert_close(edge_index_orig, edge_index_norm)

        # Check normalization: each graph should be normalized by its own max
        if edge_weight_norm is not None:
            assert edge_weight_norm.abs().max() <= 1.0 + 1e-6

            # Verify per-graph normalization
            for graph_id in range(batch_pooled.max().item() + 1):
                graph_mask = batch_pooled[edge_index_norm[0]] == graph_id
                if graph_mask.any():
                    graph_weights_norm = edge_weight_norm[graph_mask]
                    graph_weights_orig = edge_weight_orig[graph_mask]
                    expected_norm = graph_weights_orig / graph_weights_orig.abs().max()
                    torch.testing.assert_close(
                        graph_weights_norm, expected_norm, atol=1e-6, rtol=1e-6
                    )

    def test_sparse_connect_single_graph_batch(self):
        """Test SparseConnect with single graph (batch_size=1)."""
        batch_data = self.create_single_graph_batch()
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        connector = SparseConnect(edge_weight_norm=True)
        edge_index, edge_weight = connector(
            batch_data.edge_index,
            so,
            edge_weight=batch_data.edge_weight,
            batch_pooled=batch_pooled,
        )

        if edge_weight is not None:
            # Should normalize by global max for single graph
            assert edge_weight.abs().max() <= 1.0 + 1e-6

    def test_sparse_connect_repr(self):
        """Test SparseConnect __repr__ includes edge_weight_norm."""
        connector = SparseConnect(edge_weight_norm=True)
        repr_str = repr(connector)
        assert "edge_weight_norm=True" in repr_str

    def test_sparse_connect_degree_norm(self):
        """Test SparseConnect degree_norm feature."""
        batch_data = self.create_batch_graphs()
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # Test without degree normalization
        connector = SparseConnect(degree_norm=False)
        edge_index_orig, edge_weight_orig = connector(
            batch_data.edge_index, so, edge_weight=batch_data.edge_weight
        )

        # Test with degree normalization
        connector_norm = SparseConnect(degree_norm=True)
        edge_index_norm, edge_weight_norm = connector_norm(
            batch_data.edge_index, so, edge_weight=batch_data.edge_weight
        )

        # Check that structure is preserved
        torch.testing.assert_close(edge_index_orig, edge_index_norm)

        # Check degree normalization applied (weights should be different)
        if edge_weight_orig is not None and edge_weight_norm is not None:
            assert not torch.allclose(edge_weight_orig, edge_weight_norm), (
                "Edge weights should be different after degree normalization"
            )

            # Degree normalization should not increase maximum weight dramatically
            # (this is a sanity check, not a strict mathematical requirement)
            assert not torch.isnan(edge_weight_norm).any()
            assert not torch.isinf(edge_weight_norm).any()

    def test_sparse_connect_degree_norm_no_batch_pooled(self):
        """Test SparseConnect degree_norm works without batch_pooled."""
        batch_data = self.create_single_graph_batch()
        so, _ = self.get_select_output_and_batch_pooled(batch_data)

        # degree_norm should work without batch_pooled (unlike edge_weight_norm)
        connector = SparseConnect(degree_norm=True)
        edge_index, edge_weight = connector(
            batch_data.edge_index,
            so,
            edge_weight=batch_data.edge_weight,
            batch_pooled=None,  # This should work for degree_norm
        )

        if edge_weight is not None:
            assert not torch.isnan(edge_weight).any()
            assert not torch.isinf(edge_weight).any()

    def test_sparse_connect_degree_norm_and_normalize_combined(self):
        """Test SparseConnect with both degree_norm and edge_weight_norm."""
        batch_data = self.create_batch_graphs()
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # Test with both features enabled
        connector = SparseConnect(degree_norm=True, edge_weight_norm=True)
        edge_index, edge_weight = connector(
            batch_data.edge_index,
            so,
            edge_weight=batch_data.edge_weight,
            batch_pooled=batch_pooled,
        )

        if edge_weight is not None:
            # Both transformations should be applied
            assert edge_weight.abs().max() <= 1.0 + 1e-6  # edge_weight_norm
            assert not torch.isnan(edge_weight).any()
            assert not torch.isinf(edge_weight).any()

    def test_sparse_connect_degree_norm_repr(self):
        """Test SparseConnect __repr__ includes degree_norm."""
        connector = SparseConnect(degree_norm=True)
        repr_str = repr(connector)
        assert "degree_norm=True" in repr_str

        # Test with both features
        connector_both = SparseConnect(degree_norm=True, edge_weight_norm=True)
        repr_str_both = repr(connector_both)
        assert "degree_norm=True" in repr_str_both
        assert "edge_weight_norm=True" in repr_str_both

    # ===== DENSE CONNECT TESTS =====

    def test_dense_connect_with_batched_adjacency(self):
        """Test DenseConnect with 3D batched adjacency matrices."""
        # Create simple test data for dense connect
        batch_size = 2
        max_nodes = 4
        num_supernodes = 2

        # Create dense adjacency matrices with known values
        adj_dense = torch.zeros(batch_size, max_nodes, max_nodes)

        # Graph 1: simple adjacency matrix with max value 8.0
        adj_dense[0, 0, 1] = 2.0
        adj_dense[0, 1, 0] = 2.0
        adj_dense[0, 1, 2] = 8.0  # max value for graph 1
        adj_dense[0, 2, 1] = 8.0
        adj_dense[0, 2, 3] = 4.0
        adj_dense[0, 3, 2] = 4.0

        # Graph 2: simple adjacency matrix with max value 6.0
        adj_dense[1, 0, 1] = 3.0
        adj_dense[1, 1, 0] = 3.0
        adj_dense[1, 1, 2] = 6.0  # max value for graph 2
        adj_dense[1, 2, 1] = 6.0
        adj_dense[1, 2, 3] = 1.0
        adj_dense[1, 3, 2] = 1.0

        # Create dense assignment matrices (batch_size, max_nodes, num_supernodes)
        s_batched = torch.zeros(batch_size, max_nodes, num_supernodes)

        # Graph 1: nodes 0,1 -> supernode 0, nodes 2,3 -> supernode 1
        s_batched[0, 0, 0] = 1.0
        s_batched[0, 1, 0] = 1.0
        s_batched[0, 2, 1] = 1.0
        s_batched[0, 3, 1] = 1.0

        # Graph 2: nodes 0,1 -> supernode 0, nodes 2,3 -> supernode 1
        s_batched[1, 0, 0] = 1.0
        s_batched[1, 1, 0] = 1.0
        s_batched[1, 2, 1] = 1.0
        s_batched[1, 3, 1] = 1.0

        # Test without normalization
        connector = DenseConnect(edge_weight_norm=False)
        adj_orig, _ = connector(adj_dense, type("SO", (), {"s": s_batched})())

        # Test with normalization
        connector_norm = DenseConnect(edge_weight_norm=True)
        adj_norm, _ = connector_norm(adj_dense, type("SO", (), {"s": s_batched})())

        # Check that normalization happened per graph
        assert not torch.isnan(adj_norm).any(), "No NaN values should be present"
        assert not torch.isinf(adj_norm).any(), "No Inf values should be present"
        assert adj_norm.abs().max() <= 1.0 + 1e-6

        # Each graph (batch dimension) should be normalized independently
        for batch_idx in range(batch_size):
            graph_adj_orig = adj_orig[batch_idx]
            graph_adj_norm = adj_norm[batch_idx]
            max_val = graph_adj_orig.abs().max()
            if max_val > 0:
                expected_norm = graph_adj_orig / max_val
                torch.testing.assert_close(
                    graph_adj_norm, expected_norm, atol=1e-6, rtol=1e-6
                )

    def test_dense_connect_repr(self):
        """Test DenseConnect __repr__ includes edge_weight_norm."""
        connector = DenseConnect(edge_weight_norm=True)
        repr_str = repr(connector)
        assert "edge_weight_norm=True" in repr_str

    # ===== DENSE CONNECT SPT TESTS =====

    def test_dense_connect_spt_assertion_error(self):
        """Test that DenseConnectSPT raises AssertionError when batch_pooled is None."""
        batch_data = self.create_batch_graphs()
        so, _ = self.get_select_output_and_batch_pooled(batch_data)

        connector = DenseConnectSPT(edge_weight_norm=True)

        with pytest.raises(AssertionError, match="batch_pooled parameter is required"):
            connector(
                batch_data.edge_index, batch_data.edge_weight, so, batch_pooled=None
            )

    def test_dense_connect_spt_with_batch_pooled(self):
        """Test DenseConnectSPT with proper batch_pooled parameter."""
        batch_data = self.create_batch_graphs()
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # Test without normalization
        connector = DenseConnectSPT(edge_weight_norm=False)
        adj_orig, edge_weight_orig = connector(
            edge_index=batch_data.edge_index,
            edge_weight=batch_data.edge_weight,
            batch=batch_data.batch,
            so=so,
        )

        # Test with normalization
        connector_norm = DenseConnectSPT(edge_weight_norm=True)
        adj_norm, edge_weight_norm = connector_norm(
            edge_index=batch_data.edge_index,
            edge_weight=batch_data.edge_weight,
            batch=batch_data.batch,
            so=so,
            batch_pooled=batch_pooled,
        )

        adj_orig = SparseTensor.from_edge_index(
            edge_index=adj_orig, edge_attr=edge_weight_orig
        )
        adj_norm = SparseTensor.from_edge_index(
            edge_index=adj_norm, edge_attr=edge_weight_norm
        )
        # Check normalization
        if isinstance(adj_norm, SparseTensor):
            _, _, values = adj_norm.coo()
            if len(values) > 0:
                assert values.abs().max() <= 1.0 + 1e-6

                # Verify per-graph normalization
                for graph_id in range(batch_pooled.max().item() + 1):
                    # Get original values for this graph
                    _, _, orig_values = adj_orig.coo()
                    row_orig, _, _ = adj_orig.coo()
                    graph_mask_orig = batch_pooled[row_orig] == graph_id

                    if graph_mask_orig.any():
                        graph_values_orig = orig_values[graph_mask_orig]
                        row_norm, _, _ = adj_norm.coo()
                        graph_mask_norm = batch_pooled[row_norm] == graph_id
                        graph_values_norm = values[graph_mask_norm]

                        max_val = graph_values_orig.abs().max()
                        if max_val > 0:
                            expected_norm = graph_values_orig / max_val
                            torch.testing.assert_close(
                                graph_values_norm, expected_norm, atol=1e-5, rtol=1e-5
                            )

    def test_dense_connect_spt_repr(self):
        """Test DenseConnectSPT __repr__ includes edge_weight_norm."""
        connector = DenseConnectSPT(edge_weight_norm=True)
        repr_str = repr(connector)
        assert "edge_weight_norm=True" in repr_str

    # ===== EDGE CASES =====

    def test_zero_edge_weights(self):
        """Test normalization with zero edge weights (division by zero)."""
        batch_data = self.create_batch_graphs()
        batch_data.edge_weight = torch.zeros_like(batch_data.edge_weight)
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # SparseConnect
        connector = SparseConnect(edge_weight_norm=True)
        edge_index, edge_weight = connector(
            batch_data.edge_index,
            so,
            edge_weight=batch_data.edge_weight,
            batch_pooled=batch_pooled,
        )

        if edge_weight is not None:
            assert not torch.isnan(edge_weight).any()
            assert not torch.isinf(edge_weight).any()
            torch.testing.assert_close(edge_weight, torch.zeros_like(edge_weight))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
