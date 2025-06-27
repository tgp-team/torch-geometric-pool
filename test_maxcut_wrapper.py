#!/usr/bin/env python3
"""Simple test for MaxCut pooler wrapper."""

import torch

from tgp.poolers import get_pooler


def test_maxcut_wrapper():
    """Test the MaxCut wrapper functionality."""
    print("Testing MaxCut wrapper...")

    # Create simple test data
    x = torch.randn(10, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [1, 0, 3, 2, 0, 4, 2, 1, 4, 3]],
        dtype=torch.long,
    )

    try:
        # Test instantiation
        pooler = get_pooler("maxcut", in_channels=16, ratio=0.5, beta=1.0)
        print("✓ MaxCut pooler instantiation successful")

        # Test forward pass
        output = pooler(x, edge_index)
        print("✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.x.shape if output.x is not None else None}")
        print(f"  Has loss: {output.has_loss}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_maxcut_wrapper()
    if success:
        print("🎉 MaxCut wrapper test passed!")
    else:
        print("❌ MaxCut wrapper test failed!")
