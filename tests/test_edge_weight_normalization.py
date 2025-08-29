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

        connector = SparseConnect(normalize_edge_weight=True)

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
        connector = SparseConnect(normalize_edge_weight=False)
        edge_index_orig, edge_weight_orig = connector(
            batch_data.edge_index, so, edge_weight=batch_data.edge_weight
        )

        # Test with normalization
        connector_norm = SparseConnect(normalize_edge_weight=True)
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

        connector = SparseConnect(normalize_edge_weight=True)
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
        """Test SparseConnect __repr__ includes normalize_edge_weight."""
        connector = SparseConnect(normalize_edge_weight=True)
        repr_str = repr(connector)
        assert "normalize_edge_weight=True" in repr_str

    # ===== DENSE CONNECT TESTS =====

    def test_dense_connect_with_batched_adjacency(self):
        """Test DenseConnect with 3D batched adjacency matrices."""
        batch_data = self.create_batch_graphs()

        # Create dense adjacency matrices
        adj_dense = self.create_dense_adjacency(batch_data)

        # Get SelectOutput (need dense assignment matrix)
        selector = TopkSelect(in_channels=batch_data.x.size(1), ratio=0.5)
        so = selector(batch_data.x, batch=batch_data.batch)

        # Convert sparse assignment to dense batched format
        s_dense = so.s.to_dense()
        batch_size = batch_data.batch.max().item() + 1
        max_nodes = adj_dense.size(1)
        max_supernodes = so.num_supernodes

        # Create batched dense assignment matrix
        s_batched = torch.zeros(batch_size, max_nodes, max_supernodes)
        start_node = 0
        start_supernode = 0

        for batch_idx in range(batch_size):
            graph_nodes = (batch_data.batch == batch_idx).sum().item()
            graph_supernodes = (
                (BaseReduce.reduce_batch(so, batch_data.batch) == batch_idx)
                .sum()
                .item()
            )

            s_batched[
                batch_idx,
                :graph_nodes,
                start_supernode : start_supernode + graph_supernodes,
            ] = s_dense[
                start_node : start_node + graph_nodes,
                start_supernode : start_supernode + graph_supernodes,
            ]

            start_node += graph_nodes
            start_supernode += graph_supernodes

        # Test without normalization
        connector = DenseConnect(normalize_edge_weight=False)
        adj_orig, _ = connector(adj_dense, type("SO", (), {"s": s_batched})())

        # Test with normalization
        connector_norm = DenseConnect(normalize_edge_weight=True)
        adj_norm, _ = connector_norm(adj_dense, type("SO", (), {"s": s_batched})())

        # Check that normalization happened per graph
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
        """Test DenseConnect __repr__ includes normalize_edge_weight."""
        connector = DenseConnect(normalize_edge_weight=True)
        repr_str = repr(connector)
        assert "normalize_edge_weight=True" in repr_str

    # ===== DENSE CONNECT SPT TESTS =====

    def test_dense_connect_spt_assertion_error(self):
        """Test that DenseConnectSPT raises AssertionError when batch_pooled is None."""
        batch_data = self.create_batch_graphs()
        so, _ = self.get_select_output_and_batch_pooled(batch_data)

        connector = DenseConnectSPT(normalize_edge_weight=True)

        with pytest.raises(AssertionError, match="batch_pooled parameter is required"):
            connector(
                batch_data.edge_index, batch_data.edge_weight, so, batch_pooled=None
            )

    def test_dense_connect_spt_with_batch_pooled(self):
        """Test DenseConnectSPT with proper batch_pooled parameter."""
        batch_data = self.create_batch_graphs()
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # Test without normalization
        connector = DenseConnectSPT(normalize_edge_weight=False)
        adj_orig, _ = connector(batch_data.edge_index, batch_data.edge_weight, so)

        # Test with normalization
        connector_norm = DenseConnectSPT(normalize_edge_weight=True)
        adj_norm, _ = connector_norm(
            batch_data.edge_index, batch_data.edge_weight, so, batch_pooled=batch_pooled
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
        """Test DenseConnectSPT __repr__ includes normalize_edge_weight."""
        connector = DenseConnectSPT(normalize_edge_weight=True)
        repr_str = repr(connector)
        assert "normalize_edge_weight=True" in repr_str

    # ===== EDGE CASES =====

    def test_zero_edge_weights(self):
        """Test normalization with zero edge weights (division by zero)."""
        batch_data = self.create_batch_graphs()
        batch_data.edge_weight = torch.zeros_like(batch_data.edge_weight)
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # SparseConnect
        connector = SparseConnect(normalize_edge_weight=True)
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

    def test_negative_edge_weights(self):
        """Test normalization preserves signs with negative edge weights."""
        batch_data = self.create_batch_graphs()
        batch_data.edge_weight = torch.tensor(
            [-2.0, 4.0, -6.0, 1.0, -8.0, 3.0, -10.0, 2.0, 5.0]
        )
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        connector = SparseConnect(normalize_edge_weight=True)
        edge_index, edge_weight = connector(
            batch_data.edge_index,
            so,
            edge_weight=batch_data.edge_weight,
            batch_pooled=batch_pooled,
        )

        if edge_weight is not None:
            # Should preserve signs and normalize by absolute maximum
            assert edge_weight.abs().max() <= 1.0 + 1e-6
            # Check that signs are preserved (if original had negatives, normalized should too)
            orig_has_negative = (batch_data.edge_weight < 0).any()
            norm_has_negative = (edge_weight < 0).any()
            if orig_has_negative:
                assert norm_has_negative

    def test_consistency_across_methods(self):
        """Test that all methods show consistent normalization behavior."""
        batch_data = (
            self.create_single_graph_batch()
        )  # Use single graph for consistency
        so, batch_pooled = self.get_select_output_and_batch_pooled(batch_data)

        # Test SparseConnect
        sparse_connector = SparseConnect(normalize_edge_weight=True)
        _, sparse_weights = sparse_connector(
            batch_data.edge_index,
            so,
            edge_weight=batch_data.edge_weight,
            batch_pooled=batch_pooled,
        )

        # Test DenseConnectSPT
        spt_connector = DenseConnectSPT(normalize_edge_weight=True)
        spt_adj, _ = spt_connector(
            batch_data.edge_index, batch_data.edge_weight, so, batch_pooled=batch_pooled
        )

        # Both should normalize to similar ranges
        if sparse_weights is not None:
            assert sparse_weights.abs().max() <= 1.0 + 1e-6

        if isinstance(spt_adj, SparseTensor):
            _, _, spt_values = spt_adj.coo()
            if len(spt_values) > 0:
                assert spt_values.abs().max() <= 1.0 + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
