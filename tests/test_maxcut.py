"""
Comprehensive test suite for MaxCut pooling implementation.

Tests all components systematically:
1. MaxCut loss function  
2. MaxCutScoreNet
3. MaxCutSelect (both assignment modes)
4. MaxCutPooling (both assignment modes)
"""

import pytest
import torch
import math
from torch import Tensor
from torch_sparse import SparseTensor

from tgp.select.maxcut_select import MaxCutScoreNet, MaxCutSelect
from tgp.poolers.maxcut import MaxCutPooling
from tgp.utils.losses import maxcut_loss


@pytest.fixture(scope="module")
def simple_graph():
    """Create a simple test graph for consistent testing."""
    torch.manual_seed(42)
    
    # Create a small ring graph: 0-1-2-3-4-5-0
    N, F = 6, 8
    x = torch.randn(N, F)
    
    # Ring connectivity (undirected)
    edges = [(i, (i + 1) % N) for i in range(N)]
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges]).t().contiguous()
    edge_weight = torch.ones(edge_index.size(1))
    batch = torch.zeros(N, dtype=torch.long)
    
    return x, edge_index, edge_weight, batch


@pytest.fixture(scope="module")
def batched_graph():
    """Create a batched graph with two components."""
    torch.manual_seed(123)
    
    # First graph: triangle (3 nodes)
    N1, F = 3, 4
    x1 = torch.randn(N1, F)
    edges1 = [(0, 1), (1, 2), (2, 0)]
    edge_index1 = torch.tensor(edges1 + [(j, i) for i, j in edges1]).t().contiguous()
    batch1 = torch.zeros(N1, dtype=torch.long)
    
    # Second graph: chain (4 nodes, offset by N1)
    N2 = 4
    x2 = torch.randn(N2, F)
    edges2 = [(N1 + i, N1 + i + 1) for i in range(N2 - 1)]
    edge_index2 = torch.tensor(edges2 + [(j, i) for i, j in edges2]).t().contiguous()
    batch2 = torch.ones(N2, dtype=torch.long)
    
    # Combine
    x = torch.cat([x1, x2], dim=0)
    edge_index = torch.cat([edge_index1, edge_index2], dim=1)
    edge_weight = torch.ones(edge_index.size(1))
    batch = torch.cat([batch1, batch2], dim=0)
    
    return x, edge_index, edge_weight, batch


class TestMaxCutLoss:
    """Test MaxCut loss computation."""
    
    def test_maxcut_loss_basic(self, simple_graph):
        """Test basic MaxCut loss computation."""
        x, edge_index, edge_weight, batch = simple_graph
        N = x.size(0)
        
        # Create test scores
        scores = torch.randn(N)
        
        # Compute loss
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            batch_reduction="mean"
        )
        
        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
    
    def test_maxcut_loss_no_edge_weights(self, simple_graph):
        """Test MaxCut loss without edge weights."""
        x, edge_index, _, batch = simple_graph
        N = x.size(0)
        
        scores = torch.randn(N)
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=None,  # No edge weights
            batch=batch,
            batch_reduction="mean"
        )
        
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
    
    def test_maxcut_loss_batched(self, batched_graph):
        """Test MaxCut loss with batched graphs."""
        x, edge_index, edge_weight, batch = batched_graph
        N = x.size(0)
        
        scores = torch.randn(N)
        
        # Test mean reduction
        loss_mean = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            batch_reduction="mean"
        )
        
        # Test sum reduction
        loss_sum = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            batch_reduction="sum"
        )
        
        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_sum, torch.Tensor)
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)
    
    def test_maxcut_loss_gradient_flow(self, simple_graph):
        """Test that gradients flow through MaxCut loss."""
        x, edge_index, edge_weight, batch = simple_graph
        N = x.size(0)
        
        scores = torch.randn(N, requires_grad=True)
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch
        )
        
        loss.backward()
        
        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()


class TestMaxCutScoreNet:
    """Test MaxCutScoreNet component."""
    
    def test_score_net_initialization(self):
        """Test score network initialization with different parameters."""
        # Default parameters
        score_net = MaxCutScoreNet(in_channels=8)
        assert hasattr(score_net, 'mp_convs')
        assert hasattr(score_net, 'mlp')
        assert hasattr(score_net, 'final_layer')
        assert score_net.delta == 2.0
        
        # Custom parameters
        score_net_custom = MaxCutScoreNet(
            in_channels=16,
            mp_units=[32, 16],
            mlp_units=[8],
            delta=1.5
        )
        assert len(score_net_custom.mp_convs) == 2
        assert len(score_net_custom.mlp) == 1
        assert score_net_custom.delta == 1.5
    
    def test_score_net_forward(self, simple_graph):
        """Test score network forward pass."""
        x, edge_index, edge_weight, _ = simple_graph
        N, F = x.shape
        
        score_net = MaxCutScoreNet(
            in_channels=F,
            mp_units=[16, 8],
            mlp_units=[4]
        )
        score_net.eval()
        
        scores = score_net(x, edge_index, edge_weight)
        
        # Check output shape and range
        assert scores.shape == (N, 1)
        assert torch.all(scores >= -1.0) and torch.all(scores <= 1.0)  # tanh output
        assert torch.isfinite(scores).all()
    
    @pytest.mark.parametrize("mp_act,mlp_act", [
        ("relu", "tanh"), 
        ("tanh", "relu"),
        ("identity", "identity")
    ])
    def test_score_net_different_activations(self, simple_graph, mp_act, mlp_act):
        """Test score network with different activation functions."""
        x, edge_index, edge_weight, _ = simple_graph
        
        score_net = MaxCutScoreNet(
            in_channels=x.size(1),
            mp_units=[8],
            mlp_units=[4],
            mp_act=mp_act,
            mlp_act=mlp_act
        )
        score_net.eval()
        
        scores = score_net(x, edge_index, edge_weight)
        assert scores.shape == (x.size(0), 1)
        assert torch.isfinite(scores).all()


class TestMaxCutSelect:
    """Test MaxCutSelect component."""
    
    def test_maxcut_select_initialization(self):
        """Test MaxCutSelect initialization."""
        # Default parameters
        selector = MaxCutSelect(in_channels=16, ratio=0.5)
        assert selector.in_channels == 16
        assert selector.ratio == 0.5
        assert selector.assign_all_nodes == True  # Default
        assert hasattr(selector, 'score_net')
        
        # Custom parameters
        selector_custom = MaxCutSelect(
            in_channels=32,
            ratio=0.3,
            assign_all_nodes=False,
            mp_units=[16],
            mlp_units=[8]
        )
        assert selector_custom.assign_all_nodes == False
        assert selector_custom.ratio == 0.3
    
    def test_maxcut_select_assign_all_nodes_true(self, simple_graph):
        """Test MaxCutSelect with assign_all_nodes=True (assignment matrix mode)."""
        x, edge_index, edge_weight, batch = simple_graph
        N, F = x.shape
        
        selector = MaxCutSelect(
            in_channels=F,
            ratio=0.5,
            assign_all_nodes=True,
            mp_units=[8, 4],
            mlp_units=[4]
        )
        selector.eval()
        
        out = selector(x, edge_index, edge_weight, batch)
        
        expected_k = math.ceil(0.5 * N)  # Number of supernodes
        
        # Check assignment matrix properties
        assert out.num_nodes == N  # All original nodes
        assert out.num_clusters == expected_k  # Number of supernodes
        assert out.node_index.size(0) == N  # ALL nodes in assignment
        assert out.cluster_index.size(0) == N  # Each node has cluster assignment
        assert out.weight.size(0) == N  # Weight for each node
        
        # Check assignment validity
        assert torch.all(out.cluster_index >= 0)
        assert torch.all(out.cluster_index < expected_k)
        assert torch.all(out.node_index == torch.arange(N))
        
        # Check scores are stored
        assert hasattr(out, 'scores')
        assert out.scores.size(0) == N
    
    def test_maxcut_select_assign_all_nodes_false(self, simple_graph):
        """Test MaxCutSelect with assign_all_nodes=False (standard TopK mode)."""
        x, edge_index, edge_weight, batch = simple_graph
        N, F = x.shape
        
        selector = MaxCutSelect(
            in_channels=F,
            ratio=0.5,
            assign_all_nodes=False,
            mp_units=[8, 4],
            mlp_units=[4]
        )
        selector.eval()
        
        out = selector(x, edge_index, edge_weight, batch)
        
        expected_k = math.ceil(0.5 * N)  # Number of selected nodes
        
        # Check standard TopK properties
        assert out.num_nodes == N  # Total nodes in graph
        assert out.num_clusters == expected_k  # Number of selected nodes
        assert out.node_index.size(0) == expected_k  # Only selected nodes
        assert out.cluster_index.size(0) == expected_k  # Each selected node -> cluster
        assert out.weight.size(0) == expected_k  # Weight for selected nodes only
        
        # Check scores are stored for ALL nodes (for loss computation)
        assert hasattr(out, 'scores')
        assert out.scores.size(0) == N
    
    @pytest.mark.parametrize("ratio", [0.3, 0.6, 2])
    def test_maxcut_select_different_ratios(self, simple_graph, ratio):
        """Test MaxCutSelect with different ratios."""
        x, edge_index, edge_weight, batch = simple_graph
        N = x.size(0)
        
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=ratio,
            assign_all_nodes=True,  # Test assignment mode
            mp_units=[4],
            mlp_units=[2]
        )
        selector.eval()
        
        out = selector(x, edge_index, edge_weight, batch)
        
        if isinstance(ratio, float):
            expected_k = max(1, math.ceil(ratio * N))
        else:
            expected_k = min(ratio, N)
        
        # In assignment mode: all nodes assigned to expected_k supernodes
        assert out.num_clusters == expected_k
        assert out.node_index.size(0) == N  # All nodes in assignment
        assert torch.all(out.cluster_index < expected_k)
    
    def test_maxcut_select_none_edge_index_error(self, simple_graph):
        """Test that MaxCutSelect raises error when edge_index is None."""
        x, _, edge_weight, batch = simple_graph
        
        selector = MaxCutSelect(in_channels=x.size(1), ratio=0.5)
        
        with pytest.raises(ValueError, match="edge_index cannot be None for MaxCutSelect"):
            selector(x=x, edge_index=None, edge_weight=edge_weight, batch=batch)
    
    def test_maxcut_select_sparse_tensor_input(self, simple_graph):
        """Test MaxCutSelect with SparseTensor input."""
        x, edge_index, edge_weight, batch = simple_graph
        
        # Convert to SparseTensor
        sparse_adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)
        
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=0.5,
            assign_all_nodes=True,
            mp_units=[4],
            mlp_units=[2]
        )
        selector.eval()
        
        out = selector(x=x, edge_index=sparse_adj, batch=batch)
        
        # Should work correctly with SparseTensor input
        assert out.num_nodes == x.size(0)
        assert hasattr(out, 'scores')


class TestMaxCutPooling:
    """Test MaxCutPooling component."""
    
    def test_maxcut_pooling_initialization(self):
        """Test MaxCutPooling initialization."""
        # Default parameters
        pooler = MaxCutPooling(in_channels=16, ratio=0.5)
        assert pooler.in_channels == 16
        assert pooler.ratio == 0.5
        assert pooler.assign_all_nodes == True  # Default
        assert pooler.loss_coeff == 1.0
        assert pooler.has_loss == True
        
        # Custom parameters
        pooler_custom = MaxCutPooling(
            in_channels=32,
            ratio=0.3,
            assign_all_nodes=False,
            loss_coeff=2.0
        )
        assert pooler_custom.assign_all_nodes == False
        assert pooler_custom.loss_coeff == 2.0
    
    def test_maxcut_pooling_assign_all_nodes_true(self, simple_graph):
        """Test MaxCutPooling with assign_all_nodes=True."""
        x, edge_index, edge_weight, batch = simple_graph
        N, F = x.shape
        
        pooler = MaxCutPooling(
            in_channels=F,
            ratio=0.5,
            assign_all_nodes=True,
            mp_units=[8, 4],
            mlp_units=[4]
        )
        pooler.eval()
        
        out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
        
        expected_k = math.ceil(0.5 * N)
        
        # Check pooled output
        assert out.x.size(0) == expected_k  # Pooled to supernodes
        assert out.x.size(1) == F  # Same feature dimension
        assert out.edge_index.size(0) == 2  # Valid edge format
        assert out.batch is not None
        
        # Check SelectOutput
        assert out.so.num_nodes == N
        assert out.so.num_clusters == expected_k
        
        # Check loss computation
        assert out.has_loss
        assert 'maxcut_loss' in out.loss
        assert torch.isfinite(out.loss['maxcut_loss'])
    
    def test_maxcut_pooling_assign_all_nodes_false(self, simple_graph):
        """Test MaxCutPooling with assign_all_nodes=False."""
        x, edge_index, edge_weight, batch = simple_graph
        N, F = x.shape
        
        pooler = MaxCutPooling(
            in_channels=F,
            ratio=0.5,
            assign_all_nodes=False,
            mp_units=[8, 4],
            mlp_units=[4]
        )
        pooler.eval()
        
        out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
        
        expected_k = math.ceil(0.5 * N)
        
        # Check pooled output (should be same in both modes for final result)
        assert out.x.size(0) == expected_k
        assert out.x.size(1) == F
        
        # Check SelectOutput (different from assignment mode)
        assert out.so.num_nodes == N
        assert out.so.num_clusters == expected_k
        
        # Check loss computation
        assert out.has_loss
        assert 'maxcut_loss' in out.loss
        assert torch.isfinite(out.loss['maxcut_loss'])
    
    def test_maxcut_pooling_no_adjacency_matrix(self, simple_graph):
        """Test MaxCutPooling behavior when adj=None."""
        x, _, edge_weight, batch = simple_graph
        
        pooler = MaxCutPooling(in_channels=x.size(1), ratio=0.5)
        
        # Should handle missing adjacency gracefully
        with pytest.raises(ValueError, match="edge_index cannot be None for MaxCutSelect"):
            pooler(x=x, adj=None, edge_weight=edge_weight, batch=batch)
    
    def test_maxcut_pooling_lifting_mode(self, simple_graph):
        """Test MaxCutPooling lifting functionality."""
        x, edge_index, edge_weight, batch = simple_graph
        
        pooler = MaxCutPooling(in_channels=x.size(1), ratio=0.5)
        
        # First get a SelectOutput
        so = pooler.select(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        
        # Test lifting with valid SelectOutput
        x_lifted = pooler(x=x, adj=edge_index, so=so, lifting=True)
        assert x_lifted is not None
        
        # Test lifting with None SelectOutput should raise error
        with pytest.raises(ValueError, match="SelectOutput \\(so\\) cannot be None for lifting"):
            pooler(x=x, adj=edge_index, so=None, lifting=True)
    
    def test_maxcut_pooling_gradient_flow(self, simple_graph):
        """Test gradient flow through entire MaxCutPooling."""
        x, edge_index, edge_weight, batch = simple_graph
        
        x.requires_grad_(True)
        
        pooler = MaxCutPooling(in_channels=x.size(1), ratio=0.5, loss_coeff=1.0)
        out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
        
        # Compute total loss (pooling + auxiliary)
        total_loss = out.x.sum() + out.loss['maxcut_loss']
        total_loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_maxcut_pooling_batched_graphs(self, batched_graph):
        """Test MaxCutPooling with batched graphs."""
        x, edge_index, edge_weight, batch = batched_graph
        
        pooler = MaxCutPooling(
            in_channels=x.size(1),
            ratio=0.5,
            assign_all_nodes=True
        )
        pooler.eval()
        
        out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
        
        # Should handle batched input correctly
        assert out.x.size(0) > 0  # Some nodes pooled
        assert out.batch is not None
        assert len(torch.unique(out.batch)) <= len(torch.unique(batch))  # Same or fewer graphs
        assert torch.isfinite(out.loss['maxcut_loss'])
    
    def test_maxcut_pooling_extra_repr(self):
        """Test extra_repr_args method for debugging."""
        pooler = MaxCutPooling(
            in_channels=64,
            ratio=0.7,
            assign_all_nodes=False,
            loss_coeff=2.5
        )
        
        extra_args = pooler.extra_repr_args()
        
        assert isinstance(extra_args, dict)
        assert extra_args["in_channels"] == 64
        assert extra_args["ratio"] == 0.7
        assert extra_args["assign_all_nodes"] == False
        assert extra_args["loss_coeff"] == 2.5


# New tests for additional functionality
class TestBaseSelect:
    """Test base select functionality and assign_all_nodes method."""
    
    def test_select_output_assign_all_nodes_closest_strategy(self, simple_graph):
        """Test assign_all_nodes with closest_node strategy."""
        x, edge_index, edge_weight, batch = simple_graph
        N = x.size(0)
        
        # Create a MaxCutSelect to get initial SelectOutput
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=0.5,
            assign_all_nodes=False,  # Get TopK first
            mp_units=[4],
            mlp_units=[2]
        )
        selector.eval()
        
        # Get initial TopK selection
        topk_output = selector(x, edge_index, edge_weight, batch)
        
        # Test assign_all_nodes method (pass edge_weight as None to avoid validation issues)
        full_assignment = topk_output.assign_all_nodes(
            adj=edge_index,
            weight=None,  # Don't pass edge_weight here, it's for node weights
            max_iter=5,
            batch=batch,
            strategy="closest_node"
        )
        
        # Check that all nodes are now assigned
        assert full_assignment.num_nodes == N
        assert full_assignment.cluster_index.size(0) == N
        assert torch.all(full_assignment.cluster_index >= 0)
        assert torch.all(full_assignment.cluster_index < topk_output.num_clusters)
    
    def test_select_output_assign_all_nodes_random_strategy(self, simple_graph):
        """Test assign_all_nodes with random strategy."""
        x, edge_index, edge_weight, batch = simple_graph
        N = x.size(0)
        
        # Create a MaxCutSelect to get initial SelectOutput
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=0.33,  # Select fewer nodes
            assign_all_nodes=False,
            mp_units=[4],
            mlp_units=[2]
        )
        selector.eval()
        
        # Get initial TopK selection
        topk_output = selector(x, edge_index, edge_weight, batch)
        
        # Test assign_all_nodes with random strategy
        full_assignment = topk_output.assign_all_nodes(
            strategy="random",
            batch=batch
        )
        
        # Check that all nodes are now assigned
        assert full_assignment.num_nodes == N
        assert full_assignment.cluster_index.size(0) == N
        assert torch.all(full_assignment.cluster_index >= 0)
        assert torch.all(full_assignment.cluster_index < topk_output.num_clusters)
    
    def test_select_output_assign_all_nodes_error_cases(self, simple_graph):
        """Test error cases for assign_all_nodes method."""
        x, edge_index, edge_weight, batch = simple_graph
        
        # Create a SelectOutput
        selector = MaxCutSelect(in_channels=x.size(1), ratio=0.5, assign_all_nodes=False)
        selector.eval()
        topk_output = selector(x, edge_index, edge_weight, batch)
        
        # Test error when adj is None but strategy is closest_node
        with pytest.raises(AssertionError, match="adj must be provided for closest_node strategy"):
            topk_output.assign_all_nodes(
                adj=None,
                strategy="closest_node"
            )
        
        # Test error with invalid strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            topk_output.assign_all_nodes(
                adj=edge_index,
                strategy="invalid_strategy"
            )
    
    def test_maxcut_select_already_all_nodes_assigned(self, simple_graph):
        """Test that assign_all_nodes returns self when all nodes are already kept."""
        x, edge_index, edge_weight, batch = simple_graph
        
        # Create selector that assigns all nodes to supernodes (this ensures all are kept)
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=0.5,  # Use a normal ratio
            assign_all_nodes=True,  # This ensures all nodes are assigned
            mp_units=[4],
            mlp_units=[2]
        )
        selector.eval()
        
        output = selector(x, edge_index, edge_weight, batch)
        
        # With assign_all_nodes=True, all nodes should be in the assignment
        assert output.node_index.size(0) == x.size(0), "All nodes should be assigned with assign_all_nodes=True"
        
        # When all nodes are kept/assigned, assign_all_nodes should return the same object
        assigned_output = output.assign_all_nodes(adj=edge_index)
        assert assigned_output is output  # Should be the same object


class TestGetAssignments:
    """Test the get_assignments utility function."""
    
    def test_get_assignments_basic(self, simple_graph):
        """Test basic get_assignments functionality."""
        from tgp.utils.ops import get_assignments
        
        x, edge_index, edge_weight, batch = simple_graph
        N = x.size(0)
        
        # Select some nodes to keep
        kept_nodes = torch.tensor([0, 3])
        
        # Test assignment generation
        assignments = get_assignments(
            kept_node_indices=kept_nodes,
            edge_index=edge_index,
            max_iter=3,
            batch=batch,
            num_nodes=N
        )
        
        # Check output format
        assert assignments.size(0) == 2  # [node_indices, cluster_indices]
        assert assignments.size(1) == N  # All nodes assigned
        
        # Check that kept nodes are mapped to themselves
        kept_mask = torch.isin(assignments[0], kept_nodes)
        assert kept_mask.sum() == len(kept_nodes)
        
        # Check all assignments are valid
        assert torch.all(assignments[0] >= 0)
        assert torch.all(assignments[0] < N)
        assert torch.all(assignments[1] >= 0)
    
    def test_get_assignments_random_only(self, simple_graph):
        """Test get_assignments with max_iter=0 (random only)."""
        from tgp.utils.ops import get_assignments
        
        x, edge_index, edge_weight, batch = simple_graph
        N = x.size(0)
        
        kept_nodes = torch.tensor([1, 4])
        
        # Test random assignment only
        assignments = get_assignments(
            kept_node_indices=kept_nodes,
            edge_index=edge_index,
            max_iter=0,  # Only random assignment
            batch=batch,
            num_nodes=N
        )
        
        # Should still assign all nodes
        assert assignments.size(1) == N
        assert torch.all(assignments[0] >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 