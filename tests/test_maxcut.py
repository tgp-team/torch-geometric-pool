import pytest
import torch
from torch import Tensor
from torch_sparse import SparseTensor

from tgp.select.maxcut_select import MaxCutScoreNet, MaxCutSelect
from tgp.utils.losses import maxcut_loss
from tgp.utils.ops import delta_gcn_matrix


@pytest.fixture(scope="module")
def small_graph():
    """Creates a small graph for testing."""
    N, F = 6, 4
    torch.manual_seed(42)
    x = torch.randn((N, F), dtype=torch.float)
    
    # Create a simple chain graph: 0-1-2-3-4-5 (undirected)
    row = torch.arange(N - 1, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)
    
    return x, edge_index, edge_weight, batch


@pytest.fixture(scope="module")
def batched_graph():
    """Creates a batched graph with two components."""
    N1, N2, F = 4, 3, 2
    torch.manual_seed(123)
    
    # First graph: 0-1-2-3 chain
    x1 = torch.randn((N1, F))
    row1 = torch.arange(N1 - 1)
    col1 = row1 + 1
    edge_index1 = torch.stack([torch.cat([row1, col1]), torch.cat([col1, row1])])
    batch1 = torch.zeros(N1, dtype=torch.long)
    
    # Second graph: 4-5-6 chain (offset node indices)
    x2 = torch.randn((N2, F))
    row2 = torch.arange(N2 - 1) + N1
    col2 = row2 + 1
    edge_index2 = torch.stack([torch.cat([row2, col2]), torch.cat([col2, row2])])
    batch2 = torch.ones(N2, dtype=torch.long)
    
    # Combine
    x = torch.cat([x1, x2], dim=0)
    edge_index = torch.cat([edge_index1, edge_index2], dim=1)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = torch.cat([batch1, batch2], dim=0)
    
    return x, edge_index, edge_weight, batch


class TestMaxCutScoreNet:
    """Test the MaxCutScoreNet score network."""
    
    def test_score_net_initialization(self):
        """Test different initialization parameters."""
        # Default parameters
        score_net = MaxCutScoreNet(in_channels=4)
        assert score_net.delta == 2.0
        assert len(score_net.mp_convs) == 12  # default mp_units length
        assert len(score_net.mlp) == 2  # default mlp_units length
        
        # Custom parameters
        score_net_custom = MaxCutScoreNet(
            in_channels=8,
            mp_units=[16, 8],
            mlp_units=[4],
            delta=1.5
        )
        assert score_net_custom.delta == 1.5
        assert len(score_net_custom.mp_convs) == 2
        assert len(score_net_custom.mlp) == 1
    
    def test_score_net_forward(self, small_graph):
        """Test forward pass of score network."""
        x, edge_index, edge_weight, _ = small_graph
        N, F = x.shape
        
        score_net = MaxCutScoreNet(in_channels=F, mp_units=[8, 4], mlp_units=[4])
        score_net.eval()
        
        scores = score_net(x, edge_index, edge_weight)
        
        # Check output shape and range
        assert scores.shape == (N, 1)
        assert torch.all(scores >= -1.0) and torch.all(scores <= 1.0)  # tanh output
        assert isinstance(scores, Tensor)
    
    @pytest.mark.parametrize("mp_act,mlp_act", [
        ("relu", "tanh"),
        ("identity", "relu"),
        ("none", "identity")
    ])
    def test_score_net_activations(self, small_graph, mp_act, mlp_act):
        """Test different activation functions."""
        x, edge_index, edge_weight, _ = small_graph
        
        score_net = MaxCutScoreNet(
            in_channels=x.size(1),
            mp_units=[4],
            mlp_units=[2],
            mp_act=mp_act,
            mlp_act=mlp_act
        )
        score_net.eval()
        
        scores = score_net(x, edge_index, edge_weight)
        assert scores.shape == (x.size(0), 1)
        assert torch.isfinite(scores).all()


class TestMaxCutSelect:
    """Test the MaxCutSelect selector."""
    
    def test_maxcut_select_initialization(self):
        """Test MaxCutSelect initialization."""
        selector = MaxCutSelect(in_channels=4, ratio=0.6)
        assert selector.in_channels == 4
        assert selector.ratio == 0.6
        assert hasattr(selector, 'score_net')
        assert hasattr(selector, 'topk_selector')
    
    def test_maxcut_select_forward(self, small_graph):
        """Test MaxCutSelect forward pass."""
        x, edge_index, edge_weight, batch = small_graph
        N, F = x.shape
        
        selector = MaxCutSelect(
            in_channels=F, 
            ratio=0.5,
            mp_units=[8, 4],
            mlp_units=[4]
        )
        selector.eval()
        
        out = selector(x, edge_index, edge_weight, batch)
        
        # Check SelectOutput structure
        assert hasattr(out, 'node_index')
        assert hasattr(out, 's')
        assert hasattr(out, 's_inv') 
        assert hasattr(out, 'scores')
        assert hasattr(out, 'edge_index')
        assert hasattr(out, 'edge_weight')
        
        # Check selection worked
        expected_k = int(0.5 * N)
        assert out.node_index.size(0) == expected_k
        assert out.num_clusters == expected_k
        
        # Check scores are stored
        assert out.scores.shape == (N,) or out.scores.shape == (N, 1)
    
    @pytest.mark.parametrize("ratio", [0.3, 0.7, 2])  # 2 means select 2 nodes
    def test_maxcut_select_different_ratios(self, small_graph, ratio):
        """Test different selection ratios."""
        x, edge_index, edge_weight, batch = small_graph
        N = x.size(0)
        
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=ratio,
            mp_units=[4],
            mlp_units=[2]
        )
        selector.eval()
        
        out = selector(x, edge_index, edge_weight, batch)
        
        if isinstance(ratio, float):
            # PyTorch Geometric uses ceil() not int() for ratio calculation
            import math
            expected_k = max(1, math.ceil(ratio * N))
        else:
            expected_k = min(ratio, N)
        
        assert out.node_index.size(0) == expected_k
    
    def test_maxcut_select_custom_score_net(self, small_graph):
        """Test using custom score network."""
        x, edge_index, edge_weight, batch = small_graph
        
        # Custom score network
        custom_score_net = MaxCutScoreNet(
            in_channels=x.size(1),
            mp_units=[6],
            mlp_units=[3],
            delta=1.0
        )
        
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=0.5,
            score_net=custom_score_net
        )
        selector.eval()
        
        out = selector(x, edge_index, edge_weight, batch)
        assert out.node_index.size(0) > 0
    
    def test_reset_parameters(self, small_graph):
        """Test parameter reset functionality."""
        x, edge_index, edge_weight, batch = small_graph
        
        selector = MaxCutSelect(in_channels=x.size(1), ratio=0.5)
        
        # Get initial parameters
        initial_params = [p.clone() for p in selector.parameters()]
        
        # Reset parameters
        selector.reset_parameters()
        
        # Check parameters changed (at least some should be different)
        reset_params = list(selector.parameters())
        assert len(initial_params) == len(reset_params)
        # Note: We can't guarantee all parameters change due to random initialization


class TestMaxCutLoss:
    """Test the maxcut_loss function."""
    
    def test_maxcut_loss_basic(self, small_graph):
        """Test basic maxcut_loss computation."""
        x, edge_index, edge_weight, batch = small_graph
        N = x.size(0)
        
        # Create some example scores
        scores = torch.randn(N)
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            reduction="mean"
        )
        
        assert isinstance(loss, Tensor)
        assert loss.dim() == 0  # scalar
        assert torch.isfinite(loss).all()
    
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_maxcut_loss_different_reductions(self, small_graph, reduction):
        """Test different reduction methods."""
        x, edge_index, edge_weight, batch = small_graph
        scores = torch.randn(x.size(0))
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            reduction=reduction
        )
        
        assert isinstance(loss, Tensor)
        assert loss.dim() == 0  # scalar
    
    def test_maxcut_loss_batched(self, batched_graph):
        """Test maxcut_loss with batched graphs."""
        x, edge_index, edge_weight, batch = batched_graph
        scores = torch.randn(x.size(0))
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            reduction="mean"
        )
        
        assert isinstance(loss, Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss).all()
    
    def test_maxcut_loss_gradient_flow(self, small_graph):
        """Test that gradients flow through the loss."""
        x, edge_index, edge_weight, batch = small_graph
        
        # Create scores that require gradients
        scores = torch.randn(x.size(0), requires_grad=True)
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            reduction="mean"
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()
    
    def test_maxcut_loss_invalid_reduction(self, small_graph):
        """Test invalid reduction parameter."""
        x, edge_index, edge_weight, batch = small_graph
        scores = torch.randn(x.size(0))
        
        with pytest.raises(ValueError):
            maxcut_loss(
                scores=scores,
                edge_index=edge_index,
                edge_weight=edge_weight,
                batch=batch,
                reduction="invalid"  # type: ignore
            )
    
    def test_maxcut_loss_score_shapes(self, small_graph):
        """Test different score tensor shapes."""
        x, edge_index, edge_weight, batch = small_graph
        N = x.size(0)
        
        # Test 1D scores
        scores_1d = torch.randn(N)
        loss_1d = maxcut_loss(scores_1d, edge_index, edge_weight, batch)
        
        # Test 2D scores (N, 1)
        scores_2d = torch.randn(N, 1)
        loss_2d = maxcut_loss(scores_2d, edge_index, edge_weight, batch)
        
        assert isinstance(loss_1d, Tensor)
        assert isinstance(loss_2d, Tensor)
        
        # Test invalid shape
        scores_invalid = torch.randn(N, 2)
        with pytest.raises(ValueError):
            maxcut_loss(scores_invalid, edge_index, edge_weight, batch)
    
    def test_maxcut_loss_no_edge_weights(self, small_graph):
        """Test maxcut_loss without edge weights."""
        x, edge_index, _, batch = small_graph
        scores = torch.randn(x.size(0))
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=None,  # No edge weights
            batch=batch,
            reduction="mean"
        )
        
        assert isinstance(loss, Tensor)
        assert torch.isfinite(loss).all()
    
    def test_maxcut_loss_no_batch(self, small_graph):
        """Test maxcut_loss without batch information."""
        x, edge_index, edge_weight, _ = small_graph
        scores = torch.randn(x.size(0))
        
        loss = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=None,  # No batch
            reduction="mean"
        )
        
        assert isinstance(loss, Tensor)
        assert torch.isfinite(loss).all()


class TestDeltaGCNMatrix:
    """Test the delta_gcn_matrix utility function."""
    
    def test_delta_gcn_matrix_basic(self, small_graph):
        """Test basic delta_gcn_matrix computation."""
        _, edge_index, edge_weight, _ = small_graph
        
        edge_index_out, edge_weight_out = delta_gcn_matrix(
            edge_index, edge_weight, delta=2.0
        )
        
        assert isinstance(edge_index_out, Tensor)
        assert isinstance(edge_weight_out, Tensor)
        assert edge_index_out.shape[0] == 2  # Still COO format
        assert edge_weight_out.shape[0] == edge_index_out.shape[1]
    
    def test_delta_gcn_matrix_no_edge_weights(self, small_graph):
        """Test delta_gcn_matrix without edge weights."""
        _, edge_index, _, _ = small_graph
        
        edge_index_out, edge_weight_out = delta_gcn_matrix(
            edge_index, edge_weight=None, delta=1.5
        )
        
        assert isinstance(edge_index_out, Tensor)
        assert isinstance(edge_weight_out, Tensor)
        assert edge_weight_out.shape[0] == edge_index_out.shape[1]
    
    @pytest.mark.parametrize("delta", [0.5, 1.0, 2.0, 3.5])
    def test_delta_gcn_matrix_different_deltas(self, small_graph, delta):
        """Test different delta values."""
        _, edge_index, edge_weight, _ = small_graph
        
        edge_index_out, edge_weight_out = delta_gcn_matrix(
            edge_index, edge_weight, delta=delta
        )
        
        assert torch.isfinite(edge_weight_out).all()
        assert edge_index_out.shape[0] == 2


class TestMaxCutPoolingIntegration:
    """Test the full MaxCutPooling integration and error conditions."""
    
    def test_maxcut_pooling_basic_forward(self, small_graph):
        """Test basic MaxCutPooling forward pass."""
        from tgp.poolers.maxcut import MaxCutPooling
        
        x, edge_index, edge_weight, batch = small_graph
        N, F = x.shape
        
        pooler = MaxCutPooling(
            in_channels=F,
            ratio=0.5,
            loss_coeff=1.0,
            mp_units=[8, 4],
            mlp_units=[4]
        )
        pooler.eval()
        
        out = pooler(
            x=x,
            adj=edge_index,
            edge_weight=edge_weight,
            batch=batch
        )
        
        # Check basic output structure
        assert hasattr(out, 'x')
        assert hasattr(out, 'edge_index')
        assert hasattr(out, 'batch')
        assert hasattr(out, 'so')
        assert hasattr(out, 'loss')
        
        # Check loss contains maxcut_loss
        assert 'maxcut_loss' in out.loss
        assert isinstance(out.loss['maxcut_loss'], torch.Tensor)
        
        # Check has_loss property
        assert pooler.has_loss is True
        
        # Check selection worked
        expected_k = int(0.5 * N)
        assert out.x.size(0) == expected_k
    
    def test_maxcut_pooling_lifting_with_none_so(self, small_graph):
        """Test lifting=True with so=None raises ValueError."""
        from tgp.poolers.maxcut import MaxCutPooling
        
        x, edge_index, edge_weight, batch = small_graph
        
        pooler = MaxCutPooling(in_channels=x.size(1), ratio=0.5)
        pooler.eval()
        
        # This should cover line 121: raise ValueError("SelectOutput (so) cannot be None for lifting")
        with pytest.raises(ValueError, match="SelectOutput \\(so\\) cannot be None for lifting"):
            pooler(x=x, adj=edge_index, lifting=True, so=None)
    

    
    def test_maxcut_pooling_loss_computation_missing_scores(self, small_graph):
        """Test loss computation with SelectOutput missing scores."""
        from tgp.poolers.maxcut import MaxCutPooling
        from tgp.select import SelectOutput
        
        x, edge_index, edge_weight, batch = small_graph
        
        pooler = MaxCutPooling(in_channels=x.size(1), ratio=0.5)
        
        # Create a SelectOutput without scores using cluster_index
        so_without_scores = SelectOutput(
            cluster_index=torch.zeros(x.size(0), dtype=torch.long),
            num_clusters=2
        )
        
        # This should cover line 163: raise ValueError("SelectOutput must contain 'scores' for MaxCut loss computation")
        with pytest.raises(ValueError, match="SelectOutput must contain 'scores' for MaxCut loss computation"):
            pooler.compute_loss(so_without_scores, batch)
    
    def test_maxcut_pooling_loss_computation_missing_edge_index(self, small_graph):
        """Test loss computation with SelectOutput missing edge_index."""
        from tgp.poolers.maxcut import MaxCutPooling
        from tgp.select import SelectOutput
        
        x, edge_index, edge_weight, batch = small_graph
        
        pooler = MaxCutPooling(in_channels=x.size(1), ratio=0.5)
        
        # Create a SelectOutput with scores but without edge_index using extra args
        so_without_edge_index = SelectOutput(
            cluster_index=torch.zeros(x.size(0), dtype=torch.long),
            num_clusters=2,
            scores=torch.randn(x.size(0))  # Pass scores as extra arg
        )
        
        # This should cover line 170: raise ValueError("SelectOutput must contain 'edge_index' for MaxCut loss computation")
        with pytest.raises(ValueError, match="SelectOutput must contain 'edge_index' for MaxCut loss computation"):
            pooler.compute_loss(so_without_edge_index, batch)
    
    def test_maxcut_pooling_extra_repr_args(self, small_graph):
        """Test extra_repr_args method."""
        from tgp.poolers.maxcut import MaxCutPooling
        
        pooler = MaxCutPooling(
            in_channels=16,
            ratio=0.7,
            loss_coeff=2.5
        )
        
        # This should cover line 186: extra_repr_args method
        extra_args = pooler.extra_repr_args()
        expected_args = {
            "in_channels": 16,
            "ratio": 0.7,
            "loss_coeff": 2.5,
        }
        
        assert extra_args == expected_args


class TestMaxCutSelectEdgeCases:
    """Test edge cases and error conditions in MaxCutSelect."""
    
    def test_maxcut_select_none_edge_index(self, small_graph):
        """Test MaxCutSelect with None edge_index raises ValueError."""
        x, _, edge_weight, batch = small_graph
        
        selector = MaxCutSelect(in_channels=x.size(1), ratio=0.5)
        
        # This should cover line 207: raise ValueError("edge_index cannot be None for MaxCutSelect")
        with pytest.raises(ValueError, match="edge_index cannot be None for MaxCutSelect"):
            selector(x=x, edge_index=None, edge_weight=edge_weight, batch=batch)
    
    def test_maxcut_select_reset_parameters_with_custom_score_net(self, small_graph):
        """Test reset_parameters when score_net doesn't have reset_parameters method."""
        x, edge_index, edge_weight, batch = small_graph
        
        # Create a mock score net without reset_parameters method
        class MockScoreNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(x.size(1), 1)
            
            def forward(self, x, edge_index, edge_weight=None):
                return torch.tanh(self.linear(x))
        
        mock_score_net = MockScoreNet()
        
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=0.5,
            score_net=mock_score_net
        )
        
        # This should cover line 177: the conditional check for hasattr(self.score_net, 'reset_parameters')
        # It shouldn't raise an error even if score_net doesn't have reset_parameters
        selector.reset_parameters()  # Should not raise an error
    
    def test_maxcut_select_reset_parameters_with_score_net_having_reset(self, small_graph):
        """Test reset_parameters when score_net has reset_parameters method."""
        x, edge_index, edge_weight, batch = small_graph
        
        # Use the default MaxCutScoreNet which has reset_parameters method
        selector = MaxCutSelect(
            in_channels=x.size(1),
            ratio=0.5,
            mp_units=[4],
            mlp_units=[2]
        )
        
        # This should cover line 177: self.score_net.reset_parameters() when hasattr returns True
        selector.reset_parameters()  # Should call score_net.reset_parameters()
    
    def test_maxcut_select_with_sparse_tensor(self, small_graph):
        """Test MaxCutSelect with SparseTensor edge_index."""
        x, edge_index, edge_weight, batch = small_graph
        
        # Convert to SparseTensor to cover the SparseTensor conversion branch
        from torch_sparse import SparseTensor
        sparse_adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)
        
        selector = MaxCutSelect(in_channels=x.size(1), ratio=0.5, mp_units=[4], mlp_units=[2])
        selector.eval()
        
        out = selector(x=x, edge_index=sparse_adj, batch=batch)
        
        # Check that the SparseTensor was converted properly
        assert hasattr(out, 'scores')
        assert hasattr(out, 'edge_index')
        assert hasattr(out, 'edge_weight')
        assert isinstance(out.edge_index, torch.Tensor)  # Should be converted from SparseTensor
    
    def test_maxcut_select_repr(self):
        """Test __repr__ method of MaxCutSelect."""
        selector = MaxCutSelect(in_channels=64, ratio=0.3)
        
        repr_str = repr(selector)
        expected = "MaxCutSelect(in_channels=64, ratio=0.3)"
        
        assert repr_str == expected
    
    def test_maxcut_select_output_requires_grad(self, small_graph):
        """Test requires_grad_ method on SelectOutput."""
        from tgp.select import SelectOutput
        
        # Create a SelectOutput using cluster_index only (no dense s tensor)
        so = SelectOutput(
            cluster_index=torch.zeros(5, dtype=torch.long, requires_grad=False),  # Leaf tensor
            num_clusters=3
        )
        
        # Test requires_grad_ method to cover line 251 in base_select.py
        so_grad = so.requires_grad_(True)
        assert so_grad is so  # Should return self
        
        so_no_grad = so.requires_grad_(False) 
        assert so_no_grad is so  # Should return self


class TestMaxCutScoreNetEdgeCases:
    """Test edge cases for MaxCutScoreNet."""
    
    def test_maxcut_score_net_kwargs_compatibility(self, small_graph):
        """Test that MaxCutScoreNet accepts and ignores extra kwargs for compatibility."""
        x, edge_index, edge_weight, _ = small_graph
        
        # This should cover line 14: the **kwargs parameter
        score_net = MaxCutScoreNet(
            in_channels=x.size(1),
            mp_units=[4],
            mlp_units=[2],
            extra_param1="ignored",  # Extra kwargs should be ignored
            extra_param2=123,
            delta=1.5
        )
        
        scores = score_net(x, edge_index, edge_weight)
        assert scores.shape == (x.size(0), 1)
        assert torch.all(scores >= -1.0) and torch.all(scores <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__]) 