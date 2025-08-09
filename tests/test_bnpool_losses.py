import pytest
import torch
from torch.distributions import Beta

from tgp.utils.losses import (
    cluster_connectivity_prior_loss,
    kl_loss,
    sparse_bce_reconstruction_loss,
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
    adj = (((adj + adj.transpose(-1, -2)) / 2) > 0).float()  # Make symmetric
    mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    return x, adj, mask


@pytest.fixture
def variable_size_batch_data(set_random_seed):
    """Create batch data with variable graph sizes."""
    batch_size, max_nodes = 3, 5
    x = torch.randn(batch_size, max_nodes, 2)
    adj = torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float()
    adj = (((adj + adj.transpose(-1, -2)) / 2) > 0).float()  # Make symmetric

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

        # Test that masking actually affects the result
        loss_no_mask = kl_loss(q, p)
        # Results should be different when masking is applied
        assert not torch.allclose(loss, loss_no_mask, atol=1e-6)

    def test_with_batch(self, set_random_seed):
        """Test KL loss with node masking."""
        batch_size, tot_num_nodes, n_components = 5, 25, 3
        batch = torch.randint(high=batch_size, size=(tot_num_nodes,))

        # Create distributions
        alpha = torch.ones(tot_num_nodes, n_components) + 0.5
        beta = torch.ones(tot_num_nodes, n_components) + 0.5
        q = Beta(alpha, beta)

        alpha_prior = torch.ones(n_components)
        beta_prior = torch.ones(n_components) * 2.0
        p = Beta(alpha_prior, beta_prior)

        # Compute KL loss with batch
        loss = kl_loss(q, p, batch=batch, batch_size=batch_size)
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
        assert isinstance(norm_const, int)
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
        assert loss.item() <= 1e-5  # BCE loss is almost zero
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


if __name__ == "__main__":
    pytest.main([__file__])
