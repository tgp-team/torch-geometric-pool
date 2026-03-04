import pytest
import torch
from torch.distributions import Beta

from tests.test_utils import (
    make_small_batch_data,
    make_variable_size_batch_data,
)
from tests.test_utils import (
    set_random_seed as _set_random_seed,
)
from tgp.poolers.bnpool import BNPool
from tgp.utils.losses import (
    cluster_connectivity_prior_loss,
    kl_loss,
    sparse_bce_reconstruction_loss,
    weighted_bce_reconstruction_loss,
)


def test_bnpool_initialization():
    """Test BNPool initialization with different parameters."""
    # Test valid initialization
    pooler = BNPool(in_channels=4, k=3)
    assert pooler.k == 3
    assert pooler.alpha_DP == 1.0

    # Test custom parameters
    pooler = BNPool(
        in_channels=4, k=3, alpha_DP=2.0, K_var=0.5, K_mu=5.0, K_init=0.5, eta=0.8
    )
    assert pooler.alpha_DP == 2.0
    assert pooler.K_var_val == 0.5
    assert pooler.K_mu_val == 5.0
    assert pooler.K_init_val == 0.5
    assert pooler.eta == 0.8


def test_bnpool_invalid_parameters():
    """Test BNPool initialization with invalid parameters."""
    with pytest.raises(ValueError, match="alpha_DP must be positive"):
        BNPool(in_channels=4, k=3, alpha_DP=-1.0)

    with pytest.raises(ValueError, match="K_var must be positive"):
        BNPool(in_channels=4, k=3, K_var=-1.0)

    with pytest.raises(ValueError, match="eta must be positive"):
        BNPool(in_channels=4, k=3, eta=-1.0)

    with pytest.raises(ValueError, match="max_k must be positive"):
        BNPool(in_channels=4, k=-3)


def test_bnpool_reset_parameters_restores_K_from_init_value():
    k = 4
    k_init = 0.7
    pooler = BNPool(in_channels=3, k=k, K_init=k_init)

    with torch.no_grad():
        pooler.K.add_(3.0)

    pooler.reset_parameters()

    expected = k_init * torch.eye(k) - k_init * (1 - torch.eye(k))
    torch.testing.assert_close(pooler.K.detach(), expected)


def test_bnpool_extra_repr_args_contains_core_hyperparameters():
    pooler = BNPool(
        in_channels=5,
        k=3,
        batched=False,
        alpha_DP=2.5,
        K_var=0.3,
        K_mu=1.5,
        K_init=0.4,
        eta=0.8,
        train_K=False,
        num_neg_samples=12,
    )

    args = pooler.extra_repr_args()
    assert args["batched"] is False
    assert args["alpha_DP"] == 2.5
    assert args["k_prior_variance"] == 0.3
    assert args["k_prior_mean"] == 1.5
    assert args["k_init_value"] == 0.4
    assert args["eta"] == 0.8
    assert args["train_K"] is False
    assert args["num_neg_samples"] == 12


@pytest.mark.parametrize("train_k", [True, False])
def test_bnpool_training_mode(pooler_test_graph_dense_batch, train_k):
    """Test BNPool behavior in training mode."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3, train_K=train_k)
    pooler.train()

    # Enable gradient tracking
    x.requires_grad_(True)

    # Forward pass
    out = pooler(x=x, adj=adj)

    # Check if loss components are present
    assert isinstance(out.loss, dict)
    assert "quality" in out.loss
    assert "kl" in out.loss
    assert "K_prior" in out.loss

    # Check if losses are differentiable
    total_loss = sum(out.loss.values())
    total_loss.backward()

    # Check if gradients are computed
    assert x.grad is not None
    if pooler.train_K:
        assert pooler.K.grad is not None
    else:
        assert pooler.K.grad is None


def test_bnpool_eval_mode(pooler_test_graph_dense_batch):
    x, adj = pooler_test_graph_dense_batch
    batch_size, n_nodes, n_features = x.shape
    k = 2
    pooler = BNPool(in_channels=n_features, k=k)
    pooler.eval()

    out = pooler(x=x, adj=adj)

    # Check output shapes
    assert out.x.shape[0] == batch_size, "Batch dimension should be 1 for pooled x"
    assert out.x.shape[1] == k, "Number of nodes should be equal to k"
    assert out.x.shape[2] == n_features, "Feature dimension should remain unchanged"

    assert out.edge_index.shape[0] == batch_size, (
        "Batch dimension should be 1 for edge_index"
    )
    assert out.edge_index.shape[1] == out.edge_index.shape[2] == k, (
        "Adjacency matrix size should match number of clusters k"
    )


def test_bnpool_batched_forward(pooler_test_graph_dense_batch):
    """Test BNPool with batched dense inputs."""
    x, adj = pooler_test_graph_dense_batch

    pooler = BNPool(
        in_channels=x.shape[-1],
        k=3,
    )
    out = pooler(x=x, adj=adj)
    assert out.x is not None
    assert out.edge_index is not None


def test_bnpool_lifting_operation(pooler_test_graph_dense_batch):
    """Test the lifting operation in BNPool."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3)

    # First do regular pooling to get selection output
    regular_out = pooler(x=x, adj=adj)

    # Then test lifting operation
    lifted_out = pooler(x=regular_out.x, so=regular_out.so, lifting=True)

    # Check if lifted output has same dimensions as input
    assert lifted_out.shape == x.shape


def test_bnpool_batched_dense_output_mask(pooler_test_graph_dense_batch):
    """Batched dense output: out.mask equals so.out_mask, shape [B, K_max]."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3, batched=True, sparse_output=False)
    pooler.eval()
    out = pooler(x=x, adj=adj)
    assert out.so is not None
    assert out.mask is not None
    assert torch.equal(out.mask, out.so.out_mask)
    assert out.mask.dim() == 2
    assert out.mask.shape[0] == out.x.shape[0]
    assert out.mask.shape[1] == out.x.shape[1]
    assert torch.equal(out.mask, (out.so.s.sum(dim=-2) > 0))


def test_bnpool_batched_sparse_output_no_mask(pooler_test_graph_dense_batch):
    """Batched sparse output: out.mask equals so.out_mask (so.s is 3D so mask is not None)."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3, batched=True, sparse_output=True)
    pooler.eval()
    out = pooler(x=x, adj=adj)
    assert out.so is not None
    assert out.mask is not None
    assert torch.equal(out.mask, out.so.out_mask)


def test_bnpool_unbatched_forward_dense_output(pooler_test_graph_sparse_batch_tuple):
    """Unbatched path returns dense pooled outputs when sparse_output=False."""
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse_batch_tuple
    pooler = BNPool(
        in_channels=x.shape[-1],
        k=3,
        batched=False,
        sparse_output=False,
    )
    pooler.eval()

    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
    batch_size = int(batch.max().item()) + 1

    assert out.x.dim() == 3
    assert out.x.shape[0] == batch_size
    assert out.edge_index.dim() == 3
    assert out.edge_index.shape[0] == batch_size
    assert isinstance(out.loss, dict)
    assert {"quality", "kl", "K_prior"} <= set(out.loss.keys())


def test_bnpool_unbatched_forward_sparse_output(pooler_test_graph_sparse_batch_tuple):
    """Unbatched path returns sparse pooled outputs when sparse_output=True."""
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse_batch_tuple
    pooler = BNPool(
        in_channels=x.shape[-1],
        k=3,
        batched=False,
        sparse_output=True,
    )
    pooler.eval()

    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)

    assert out.x.dim() == 2
    assert out.edge_index.dim() == 2
    assert out.edge_index.shape[0] == 2
    assert out.edge_weight is not None
    assert out.edge_weight.numel() == out.edge_index.shape[1]
    assert out.batch is not None
    assert out.batch.shape[0] == out.x.shape[0]
    assert isinstance(out.loss, dict)
    assert {"quality", "kl", "K_prior"} <= set(out.loss.keys())


def test_bnpool_compute_loss_without_mask(pooler_test_graph_dense_batch):
    """compute_loss supports mask=None and uses adjacency size as normalizer."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3, batched=True)
    pooler.eval()

    out = pooler(x=x, adj=adj)
    loss = pooler.compute_loss(adj=adj, mask=None, so=out.so)

    assert set(loss.keys()) == {"quality", "kl", "K_prior"}
    assert all(torch.is_tensor(v) for v in loss.values())
    assert all(v.dim() == 0 for v in loss.values())


def test_bnpool_compute_sparse_loss_without_batch_train_k_false(
    pooler_test_graph_sparse,
):
    """Sparse loss path with batch=None and train_K=False sets K_prior to 0."""
    x, edge_index, _, _ = pooler_test_graph_sparse
    pooler = BNPool(
        in_channels=x.shape[-1],
        k=3,
        batched=False,
        train_K=False,
    )
    pooler.eval()

    so = pooler.select(x=x, batch=None)
    loss = pooler.compute_sparse_loss(adj=edge_index, batch=None, so=so)

    assert set(loss.keys()) == {"quality", "kl", "K_prior"}
    assert torch.isclose(loss["K_prior"], torch.tensor(0.0))
    assert loss["quality"].item() >= 0
    assert loss["kl"].item() >= 0


def test_bnpool_get_sparse_rec_loss_with_batch(pooler_test_graph_sparse_batch_tuple):
    """get_sparse_rec_loss supports batched sparse inputs and returns per-graph norm constants."""
    x, edge_index, _, batch = pooler_test_graph_sparse_batch_tuple
    pooler = BNPool(in_channels=x.shape[-1], k=3, batched=False)
    pooler.eval()

    so = pooler.select(x=x, batch=batch)
    batch_size = int(batch.max().item()) + 1
    rec_loss, norm_const = pooler.get_sparse_rec_loss(
        node_assignment=so.s,
        adj=edge_index,
        batch=batch,
        batch_size=batch_size,
    )

    assert torch.is_tensor(rec_loss)
    assert rec_loss.dim() == 0
    assert rec_loss.item() >= 0
    assert torch.is_tensor(norm_const)
    assert norm_const.numel() == batch_size


def test_bnpool_get_prob_link_logit_matches_manual_computation():
    pooler = BNPool(in_channels=2, k=2, batched=False)
    node_assignment = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]], dtype=torch.float
    )
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    logits = pooler.get_prob_link_logit(
        node_assignment=node_assignment, edges_list=edges
    )
    expected = ((node_assignment[edges[0]] @ pooler.K) * node_assignment[edges[1]]).sum(
        -1
    )

    torch.testing.assert_close(logits, expected)


@pytest.fixture
def set_random_seed():
    _set_random_seed(42)


@pytest.fixture
def small_batch_data(set_random_seed):
    return make_small_batch_data(batch_size=2, n_nodes=4, n_features=3, seed=42)


@pytest.fixture
def variable_size_batch_data(set_random_seed):
    return make_variable_size_batch_data(
        batch_size=3, max_nodes=5, n_features=2, seed=42
    )


class TestWeightedBCEReconstructionLoss:
    """Test cases for weighted_bce_reconstruction_loss function."""

    def test_basic_functionality(self, small_batch_data):
        """Test basic functionality of weighted BCE reconstruction loss."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]

        # Create reconstructed adjacency matrix (logits)
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        # Test basic loss computation
        loss = weighted_bce_reconstruction_loss(rec_adj, adj, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar after reduction
        assert loss.item() >= 0  # BCE loss is non-negative

    def test_balance_links_parameter(self, small_batch_data):
        """Test the balance_links parameter effect."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        # Test with and without link balancing
        loss_balanced = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, balance_links=True
        )
        loss_unbalanced = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, balance_links=False
        )

        assert isinstance(loss_balanced, torch.Tensor)
        assert isinstance(loss_unbalanced, torch.Tensor)
        # Losses should generally be different when balancing is applied
        assert not torch.allclose(loss_balanced, loss_unbalanced, atol=1e-6)

    def test_normalize_loss_parameter(self, small_batch_data):
        """Test the normalize_loss parameter effect."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        # Test with and without normalization
        loss_normalized = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, normalizing_const=torch.tensor(2)
        )
        loss_unnormalized = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, normalizing_const=None
        )

        assert isinstance(loss_normalized, torch.Tensor)
        assert isinstance(loss_unnormalized, torch.Tensor)
        # Normalized loss should be smaller
        assert loss_normalized.item() <= loss_unnormalized.item()

    def test_with_variable_masks(self, variable_size_batch_data):
        """Test with variable graph sizes."""
        x, adj, mask = variable_size_batch_data
        batch_size, max_nodes = adj.shape[:2]
        rec_adj = torch.randn(batch_size, max_nodes, max_nodes)

        loss = weighted_bce_reconstruction_loss(rec_adj, adj, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_no_mask(self, small_batch_data):
        """Test without providing a mask."""
        x, adj, _ = small_batch_data
        batch_size, n_nodes = adj.shape[:2]
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        loss = weighted_bce_reconstruction_loss(rec_adj, adj, mask=None)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_batch_reduction_methods(self, small_batch_data):
        """Test different batch reduction methods."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        # Test different batch reduction methods
        loss_mean = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, batch_reduction="mean"
        )
        loss_sum = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, batch_reduction="sum"
        )

        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_sum, torch.Tensor)
        assert loss_sum.item() >= loss_mean.item()

    def test_perfect_reconstruction(self, small_batch_data):
        """Test perfect reconstruction scenario."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]

        # Create perfect reconstruction by using high logits
        rec_adj = adj * 10.0 + (1 - adj) * (-10.0)

        loss = weighted_bce_reconstruction_loss(rec_adj, adj, mask)

        # Perfect reconstruction should yield very low loss
        assert loss.item() < 0.1


class TestKLLoss:
    """Test cases for kl_loss function."""

    def test_initialisation(self, set_random_seed):
        with pytest.raises(ValueError) as error_info:
            # both mask and batch specified will raise an exception
            kl_loss(None, None, torch.tensor(1.0), torch.tensor(1.0), None)
        assert error_info.value.args[0] == "Cannot specify both mask and batch"

        with pytest.raises(ValueError) as error_info:
            # if batch is specified but batch_size is not, will raise an exception
            kl_loss(None, None, None, torch.tensor(1.0), None)
        assert (
            error_info.value.args[0]
            == "Batch size must be specified if batch is specified"
        )

    def test_basic_functionality(self, set_random_seed):
        """Test basic KL divergence computation."""
        batch_size, n_nodes, n_components = 2, 4, 3

        # Create distributions
        alpha = torch.ones(batch_size, n_nodes, n_components) + 0.5
        beta = torch.ones(batch_size, n_nodes, n_components) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components)
        beta_prior = torch.ones(n_components) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Compute KL loss
        loss = kl_loss(q, p)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar after reduction
        assert loss.item() >= 0  # KL divergence is non-negative

    def test_with_mask(self, set_random_seed):
        """Test KL loss with node masking."""
        batch_size, max_nodes, n_components = 2, 5, 3

        # Create distributions
        alpha = torch.ones(batch_size, max_nodes, n_components) + 0.5
        beta = torch.ones(batch_size, max_nodes, n_components) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components)
        beta_prior = torch.ones(n_components) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Create mask
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :4] = True

        # Compute KL loss with mask
        loss = kl_loss(q, p, mask=mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_multidimensional_mask_coverage(self, set_random_seed):
        """Test KL loss with multi-dimensional mask to ensure lines 637-639 are covered."""
        batch_size, max_nodes, n_components = 3, 6, 4

        # Create distributions with more complex shapes
        alpha = torch.ones(batch_size, max_nodes, n_components) + torch.rand(
            batch_size, max_nodes, n_components
        )
        beta = torch.ones(batch_size, max_nodes, n_components) + torch.rand(
            batch_size, max_nodes, n_components
        )
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components) + 0.3
        beta_prior = torch.ones(n_components) * 1.5
        p = Beta(alpha_prior, beta_prior)

        # Create a 2D mask with explicit False values to ensure not torch.all(mask) is True
        mask = torch.ones(batch_size, max_nodes, dtype=torch.bool)
        # Make some entries False to ensure torch.all(mask) is False
        mask[0, 4:] = False  # Last 2 nodes of first graph are masked
        mask[1, 5:] = False  # Last 1 node of second graph is masked
        mask[2, 3:] = False  # Last 3 nodes of third graph are masked

        # Verify our mask conditions
        assert mask.dim() == 2, "Mask should be 2D"
        assert not torch.all(mask), "Mask should have some False values"

        # This should trigger the multi-dimensional mask handling (lines 637-639)
        loss = kl_loss(q, p, mask=mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Test that masking actually affects the result
        loss_no_mask = kl_loss(q, p)
        # Results should be different when masking is applied
        assert not torch.allclose(loss, loss_no_mask, atol=1e-6)

    def test_1d_mask_coverage(self, set_random_seed):
        """Test KL loss with 1D mask to cover the False branch of mask.dim() > 1."""
        # Create a scenario where we can use a 1D mask
        n_components = 3
        n_total_nodes = 8  # Total nodes across all graphs in batch

        # Create distributions with shape (n_total_nodes, n_components)
        alpha = torch.ones(n_total_nodes, n_components) + torch.rand(
            n_total_nodes, n_components
        )
        beta = torch.ones(n_total_nodes, n_components) + torch.rand(
            n_total_nodes, n_components
        )
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components) + 0.2
        beta_prior = torch.ones(n_components) * 1.8
        p = Beta(alpha_prior, beta_prior)

        # Create a 1D mask with some False values
        mask = torch.ones(n_total_nodes, dtype=torch.bool)
        mask[5:] = False  # Mask last 3 nodes

        # Verify our mask conditions for the False branch
        assert mask.dim() == 1, "Mask should be 1D"
        assert not torch.all(mask), "Mask should have some False values"

        # This should trigger the 1D mask handling (False branch of mask.dim() > 1)
        loss = kl_loss(q, p, mask=mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Test that masking actually affects the result
        loss_no_mask = kl_loss(q, p)
        # Results should be different when masking is applied
        assert not torch.allclose(loss, loss_no_mask, atol=1e-6)

    def test_normalize_loss(self, set_random_seed):
        """Test loss normalization."""
        batch_size, n_nodes, n_components = 2, 4, 3

        # Create distributions
        alpha = torch.ones(batch_size, n_nodes, n_components) + 0.5
        beta = torch.ones(batch_size, n_nodes, n_components) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components)
        beta_prior = torch.ones(n_components) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Test with and without normalization
        loss_normalized = kl_loss(q, p, normalizing_const=torch.tensor(2))
        loss_unnormalized = kl_loss(q, p, normalizing_const=None)

        assert isinstance(loss_normalized, torch.Tensor)
        assert isinstance(loss_unnormalized, torch.Tensor)
        # Normalized loss should be smaller
        assert loss_normalized.item() <= loss_unnormalized.item()

    def test_batch_reduction_methods(self, set_random_seed):
        """Test different batch reduction methods."""
        batch_size, n_nodes, n_components = 2, 4, 3

        # Create distributions
        alpha = torch.ones(batch_size, n_nodes, n_components) + 0.5
        beta = torch.ones(batch_size, n_nodes, n_components) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components)
        beta_prior = torch.ones(n_components) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Test different batch reduction methods
        loss_mean = kl_loss(q, p, batch_reduction="mean")
        loss_sum = kl_loss(q, p, batch_reduction="sum")

        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_sum, torch.Tensor)
        assert loss_sum.item() >= loss_mean.item()

    def test_identical_distributions(self, set_random_seed):
        """Test KL loss between identical distributions."""
        batch_size, n_nodes, n_components = 2, 4, 3

        # Create identical distributions
        alpha = torch.ones(batch_size, n_nodes, n_components) + 0.5
        beta = torch.ones(batch_size, n_nodes, n_components) + 0.5
        q = Beta(alpha, beta)
        p = Beta(alpha, beta)

        loss = kl_loss(q, p)

        # KL divergence between identical distributions should be very small
        assert loss.item() < 1e-6


class TestClusterConnectivityPriorLoss:
    """Test cases for cluster_connectivity_prior_loss function."""

    def test_basic_functionality(self, set_random_seed):
        """Test basic functionality of cluster connectivity prior loss."""
        k = 3
        K = torch.randn(k, k)  # Cluster connectivity matrix
        K_mu = torch.zeros(k, k)  # Prior mean
        K_var = torch.tensor(1.0)  # Prior variance

        loss = cluster_connectivity_prior_loss(K, K_mu, K_var)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Should be non-negative

    def test_zero_loss_with_perfect_prior(self, set_random_seed):
        """Test that loss is zero when K equals the prior mean."""
        k = 3
        K_mu = torch.randn(k, k)
        K = K_mu.clone()  # Set K equal to prior mean
        K_var = torch.tensor(1.0)

        loss = cluster_connectivity_prior_loss(K, K_mu, K_var)

        # Loss should be very small when K equals prior mean
        assert loss.item() < 1e-6

    def test_loss_scaling_with_variance(self, set_random_seed):
        """Test that loss scales inversely with variance."""
        k = 3
        K = torch.randn(k, k)
        K_mu = torch.zeros(k, k)

        # Test with different variances
        K_var_small = torch.tensor(0.1)
        K_var_large = torch.tensor(10.0)

        loss_small_var = cluster_connectivity_prior_loss(K, K_mu, K_var_small)
        loss_large_var = cluster_connectivity_prior_loss(K, K_mu, K_var_large)

        # Loss should be larger with smaller variance (tighter prior)
        assert loss_small_var.item() > loss_large_var.item()

    def test_normalize_loss_parameter(self, set_random_seed):
        """Test the normalize_loss parameter."""
        k = 3
        K = torch.randn(k, k)
        K_mu = torch.zeros(k, k)
        K_var = torch.tensor(1.0)

        # Test with and without normalization
        loss_normalized = cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalizing_const=torch.tensor(2)
        )
        loss_unnormalized = cluster_connectivity_prior_loss(
            K,
            K_mu,
            K_var,
            normalizing_const=None,
        )

        assert isinstance(loss_normalized, torch.Tensor)
        assert isinstance(loss_unnormalized, torch.Tensor)
        # Normalized loss should be smaller
        assert loss_normalized.mean().item() <= loss_unnormalized.item()

    def test_batch_reduction_methods(self, set_random_seed):
        """Test different batch reduction methods when normalization is used."""
        k = 3
        K = torch.randn(k, k)
        K_mu = torch.zeros(k, k)
        K_var = torch.tensor(1.0)

        # Test different batch reduction methods
        loss_mean = cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalizing_const=torch.tensor(2), batch_reduction="mean"
        )
        loss_sum = cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalizing_const=torch.tensor(2), batch_reduction="sum"
        )

        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_sum, torch.Tensor)
        assert loss_sum.item() >= loss_mean.item()

    def test_large_deviation_increases_loss(self, set_random_seed):
        """Test that large deviations from prior increase loss."""
        k = 3
        K_mu = torch.zeros(k, k)
        K_var = torch.tensor(1.0)

        # Small deviation
        K_small = K_mu + 0.1 * torch.randn(k, k)
        loss_small = cluster_connectivity_prior_loss(K_small, K_mu, K_var)

        # Large deviation
        K_large = K_mu + 5.0 * torch.randn(k, k)
        loss_large = cluster_connectivity_prior_loss(K_large, K_mu, K_var)

        # Large deviation should result in higher loss
        assert loss_large.item() > loss_small.item()


class TestSparseBCEReconstructionLoss:
    def test_basic_functionality(self, set_random_seed):
        """Test the SparseBCEReconstructionLoss function."""
        tot_num_edges = 25
        link_prob_loigit = 2 * torch.randn((tot_num_edges,))
        true_y = torch.randint(high=2, size=(tot_num_edges,), dtype=torch.float)

        loss, norm_const = sparse_bce_reconstruction_loss(link_prob_loigit, true_y)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(norm_const, torch.Tensor)
        assert norm_const == tot_num_edges
        assert loss.dim() == 0  # Scalar after reduction
        assert loss.item() >= 0  # BCE loss is non-negative

    def test_with_batch(self, set_random_seed):
        """Test KL loss with node masking."""
        num_edges_per_graph = [7, 4, 5, 6, 3]
        batch_size, tot_num_edges = len(num_edges_per_graph), sum(num_edges_per_graph)
        link_prob_loigit = 2 * torch.randn((tot_num_edges,))
        true_y = torch.randint(high=2, size=(tot_num_edges,), dtype=torch.float)
        batch = torch.tensor(
            sum([num_edges_per_graph[i] * [i] for i in range(batch_size)], [])
        )

        loss, norm_const = sparse_bce_reconstruction_loss(
            link_prob_loigit, true_y, batch, batch_size
        )

        assert isinstance(loss, torch.Tensor)
        assert isinstance(norm_const, torch.Tensor)
        assert loss.dim() == 0  # Scalar after reduction
        assert loss.item() >= 0  # BCE loss is non-negative
        assert all(
            [
                int(norm_const[i].item()) == num_edges_per_graph[i]
                for i in range(batch_size)
            ]
        )

        loss_no_batch, _ = sparse_bce_reconstruction_loss(link_prob_loigit, true_y)

        # Results should be different when masking is applied
        assert not torch.allclose(loss, loss_no_batch, atol=1e-6)

    def test_perfect_reconstruction(self, set_random_seed):
        """Test KL loss with node masking."""
        num_edges_per_graph = [7, 4, 5, 6, 3]
        batch_size, tot_num_edges = len(num_edges_per_graph), sum(num_edges_per_graph)
        true_y = torch.randint(high=2, size=(tot_num_edges,), dtype=torch.float)
        link_prob_loigit = 10 * (2 * true_y - 1)
        batch = torch.tensor(
            sum([num_edges_per_graph[i] * [i] for i in range(batch_size)], [])
        )

        loss, norm_const = sparse_bce_reconstruction_loss(
            link_prob_loigit, true_y, batch, batch_size
        )

        assert isinstance(loss, torch.Tensor)
        assert isinstance(norm_const, torch.Tensor)
        assert loss.dim() == 0  # Scalar after reduction
        assert loss.item() <= 1e-3  # BCE loss is almost zero
        assert all(
            [
                int(norm_const[i].item()) == num_edges_per_graph[i]
                for i in range(batch_size)
            ]
        )

        loss_no_batch, _ = sparse_bce_reconstruction_loss(link_prob_loigit, true_y)

        # Results should be equal
        assert torch.allclose(loss, loss_no_batch, atol=1e-6)

    def test_batch_reduction_methods(self, small_batch_data):
        """Test different batch reduction methods."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        # Test different batch reduction methods
        loss_mean = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, batch_reduction="mean"
        )
        loss_sum = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, batch_reduction="sum"
        )

        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_sum, torch.Tensor)
        assert loss_sum.item() >= loss_mean.item()


class TestIntegrationWithBNPool:
    """Integration tests to verify losses work together."""

    def test_losses_work_together(self, small_batch_data):
        """Test that all three losses can be computed together."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]
        k = 3

        # Create test data for BNPool scenario
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        # Create distributions for KL loss
        alpha = torch.ones(batch_size, n_nodes, k - 1) + 0.5
        beta = torch.ones(batch_size, n_nodes, k - 1) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(k - 1)
        beta_prior = torch.ones(k - 1) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Create cluster connectivity matrix
        K = torch.randn(k, k)
        K_mu = torch.zeros(k, k)
        K_var = torch.tensor(1.0)

        # Compute all three losses
        recon_loss = weighted_bce_reconstruction_loss(rec_adj, adj, mask)
        kl_loss_val = kl_loss(q, p, mask=mask)
        prior_loss = cluster_connectivity_prior_loss(K, K_mu, K_var)

        # All losses should be valid
        assert isinstance(recon_loss, torch.Tensor)
        assert isinstance(kl_loss_val, torch.Tensor)
        assert isinstance(prior_loss, torch.Tensor)

        assert recon_loss.dim() == 0
        assert kl_loss_val.dim() == 0
        assert prior_loss.dim() == 0

        assert recon_loss.item() >= 0
        assert kl_loss_val.item() >= 0
        assert prior_loss.item() >= 0

        # Total loss should work
        total_loss = recon_loss + kl_loss_val + prior_loss
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
