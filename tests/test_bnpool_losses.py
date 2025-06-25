import pytest
import torch
from torch.distributions import Beta

from tgp.utils.losses import (
    cluster_connectivity_prior_loss,
    kl_loss,
    weighted_bce_reconstruction_loss,
)


@pytest.fixture
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)


@pytest.fixture
def small_batch_data(set_random_seed):
    """Create small batch data for testing."""
    batch_size, n_nodes, n_features = 2, 4, 3
    x = torch.randn(batch_size, n_nodes, n_features)
    adj = torch.randint(0, 2, (batch_size, n_nodes, n_nodes)).float()
    adj = (adj + adj.transpose(-1, -2)) / 2  # Make symmetric
    mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    return x, adj, mask


@pytest.fixture
def variable_size_batch_data(set_random_seed):
    """Create batch data with variable graph sizes."""
    batch_size, max_nodes = 3, 5
    x = torch.randn(batch_size, max_nodes, 2)
    adj = torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float()
    adj = (adj + adj.transpose(-1, -2)) / 2  # Make symmetric

    # Create variable masks
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    mask[0, :3] = True  # First graph has 3 nodes
    mask[1, :4] = True  # Second graph has 4 nodes
    mask[2, :5] = True  # Third graph has 5 nodes

    return x, adj, mask


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
            rec_adj, adj, mask, normalize_loss=True
        )
        loss_unnormalized = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, normalize_loss=False
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

    def test_reduction_methods(self, small_batch_data):
        """Test different reduction methods."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]
        rec_adj = torch.randn(batch_size, n_nodes, n_nodes)

        # Test different reduction methods
        loss_mean = weighted_bce_reconstruction_loss(
            rec_adj, adj, mask, reduction="mean"
        )
        loss_sum = weighted_bce_reconstruction_loss(rec_adj, adj, mask, reduction="sum")

        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_sum, torch.Tensor)
        assert loss_sum.item() >= loss_mean.item()

    def test_perfect_reconstruction(self, small_batch_data):
        """Test perfect reconstruction scenario."""
        x, adj, mask = small_batch_data
        batch_size, n_nodes = adj.shape[:2]

        # Create a truly binary adjacency matrix for perfect reconstruction test
        binary_adj = (adj > 0.5).float()  # Convert to strictly binary {0, 1}

        # Create perfect reconstruction by using high logits
        rec_adj = binary_adj * 10 + (1 - binary_adj) * (-10)

        loss = weighted_bce_reconstruction_loss(rec_adj, binary_adj, mask)

        # Perfect reconstruction should yield very low loss
        assert loss.item() < 0.1


class TestKLLoss:
    """Test cases for kl_loss function."""

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
        loss = kl_loss(q, p, mask=mask, node_axis=1)

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
        loss = kl_loss(q, p, mask=mask, node_axis=1)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Test that masking actually affects the result
        loss_no_mask = kl_loss(q, p, node_axis=1)
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
        loss = kl_loss(q, p, mask=mask, node_axis=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Test that masking actually affects the result
        loss_no_mask = kl_loss(q, p, node_axis=0)
        # Results should be different when masking is applied
        assert not torch.allclose(loss, loss_no_mask, atol=1e-6)

    def test_sum_axes_parameter(self, set_random_seed):
        """Test the sum_axes parameter for flexible axis control."""
        batch_size, n_nodes, n_components = 2, 4, 3

        # Create distributions
        alpha = torch.ones(batch_size, n_nodes, n_components) + 0.5
        beta = torch.ones(batch_size, n_nodes, n_components) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components)
        beta_prior = torch.ones(n_components) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Test different sum_axes configurations
        loss_default = kl_loss(q, p)
        loss_custom = kl_loss(q, p, sum_axes=[2, 1])  # Sum components first, then nodes

        assert isinstance(loss_default, torch.Tensor)
        assert isinstance(loss_custom, torch.Tensor)
        # Results should be the same for this case
        assert torch.allclose(loss_default, loss_custom, atol=1e-6)

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
        loss_normalized = kl_loss(q, p, node_axis=1, normalize_loss=True)
        loss_unnormalized = kl_loss(q, p, node_axis=1, normalize_loss=False)

        assert isinstance(loss_normalized, torch.Tensor)
        assert isinstance(loss_unnormalized, torch.Tensor)
        # Normalized loss should be smaller
        assert loss_normalized.item() <= loss_unnormalized.item()

    def test_reduction_methods(self, set_random_seed):
        """Test different reduction methods."""
        batch_size, n_nodes, n_components = 2, 4, 3

        # Create distributions
        alpha = torch.ones(batch_size, n_nodes, n_components) + 0.5
        beta = torch.ones(batch_size, n_nodes, n_components) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components)
        beta_prior = torch.ones(n_components) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Test different reduction methods
        loss_mean = kl_loss(q, p, reduction="mean")
        loss_sum = kl_loss(q, p, reduction="sum")

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

        # Create a mask for normalization
        mask = torch.ones(2, 4, dtype=torch.bool)  # Batch size 2, 4 nodes each

        # Test with and without normalization
        loss_normalized = cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalize_loss=True, mask=mask
        )
        loss_unnormalized = cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalize_loss=False, mask=mask
        )

        assert isinstance(loss_normalized, torch.Tensor)
        assert isinstance(loss_unnormalized, torch.Tensor)
        # Normalized loss should be smaller
        assert loss_normalized.mean().item() <= loss_unnormalized.item()

    def test_reduction_methods(self, set_random_seed):
        """Test different reduction methods when normalization is used."""
        k = 3
        K = torch.randn(k, k)
        K_mu = torch.zeros(k, k)
        K_var = torch.tensor(1.0)

        # Create a mask for normalization (this creates a tensor loss)
        mask = torch.ones(2, 4, dtype=torch.bool)

        # Test different reduction methods
        loss_mean = cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalize_loss=True, mask=mask, reduction="mean"
        )
        loss_sum = cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalize_loss=True, mask=mask, reduction="sum"
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
        kl_loss_val = kl_loss(q, p, mask=mask, node_axis=1)
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


if __name__ == "__main__":
    pytest.main([__file__])
