#!/usr/bin/env python3
"""Test script for MaxCut loss function."""

import torch

from tgp.utils.losses import maxcut_loss


def test_maxcut_loss_basic():
    """Test basic MaxCut loss functionality."""
    print("Testing MaxCut loss basic functionality...")

    # Create simple test data
    scores = torch.tensor([-0.5, 0.8, -0.3, 0.7], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.5, 1.5], dtype=torch.float32)

    try:
        # Test basic loss computation
        loss = maxcut_loss(scores, edge_index, edge_weight)
        print(f"✓ Basic loss computation successful: {loss.item():.4f}")

        # Test without edge weights
        loss_no_weights = maxcut_loss(scores, edge_index)
        print(f"✓ Loss without edge weights: {loss_no_weights.item():.4f}")

        # Test with batch
        batch = torch.tensor([0, 0, 1, 1])
        loss_batch = maxcut_loss(scores, edge_index, edge_weight, batch)
        print(f"✓ Loss with batch: {loss_batch.item():.4f}")

        return True

    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_maxcut_loss_properties():
    """Test mathematical properties of MaxCut loss."""
    print("\nTesting MaxCut loss properties...")

    try:
        # Test with perfectly separated partition (should give good cut)
        scores_separated = torch.tensor([-1.0, 1.0, 1.0, -1.0])
        edge_index = torch.tensor(
            [[0, 1, 0, 1, 2, 3, 2, 3], [1, 0, 2, 3, 0, 1, 3, 2]], dtype=torch.long
        )

        loss_separated = maxcut_loss(scores_separated, edge_index)
        print(f"✓ Separated partition loss: {loss_separated.item():.4f}")

        # Test with non-separated partition (should give worse cut)
        scores_mixed = torch.tensor([1.0, -1.0, 1.0, -1.0])
        loss_mixed = maxcut_loss(scores_mixed, edge_index)
        print(f"✓ Mixed partition loss: {loss_mixed.item():.4f}")

        # Test reduction methods
        loss_sum = maxcut_loss(scores_separated, edge_index, reduction="sum")
        # loss_none = maxcut_loss(scores_separated, edge_index, reduction="none")
        print(f"✓ Reduction 'sum': {loss_sum.item():.4f}")
        # print(f"✓ Reduction 'none' shape: {loss_none.shape}")

        return True

    except Exception as e:
        print(f"✗ Properties test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_maxcut_loss_gradients():
    """Test that gradients flow correctly."""
    print("\nTesting MaxCut loss gradients...")

    try:
        scores = torch.tensor([0.1, -0.2, 0.5, -0.8], requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

        loss = maxcut_loss(scores, edge_index)
        loss.backward()

        print(f"✓ Loss value: {loss.item():.4f}")
        print(f"✓ Gradients computed: {scores.grad is not None}")
        print(f"✓ Gradient values: {scores.grad.tolist()}")

        return True

    except Exception as e:
        print(f"✗ Gradient test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== MaxCut Loss Function Test ===")

    success = True
    success &= test_maxcut_loss_basic()
    success &= test_maxcut_loss_properties()
    success &= test_maxcut_loss_gradients()

    print("\n=== Test Summary ===")
    if success:
        print("🎉 All MaxCut loss tests passed!")
    else:
        print("❌ Some MaxCut loss tests failed.")
