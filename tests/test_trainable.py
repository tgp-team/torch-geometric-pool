import pytest

from tgp.poolers import get_pooler, pooler_map


def test_poolers_trainable():
    from tgp.imports import HAS_TORCH_SPARSE

    for POOLER, value in pooler_map.items():
        # Skip PANPooling if torch_sparse is not available
        if POOLER == "pan" and not HAS_TORCH_SPARSE:
            pytest.skip("PANPooling requires torch_sparse")

        PARAMS = {
            "k": 3,
            "in_channels": 16,
            "scorer": "degree",
        }

        pooler = get_pooler(POOLER, **PARAMS)

        # Non-trainable poolers
        if POOLER in ["ndp", "nmf", "graclus", "kmis", "lap", "nopool"]:
            assert not pooler.is_trainable, f"Pooler {POOLER} should not be trainable"
        else:
            assert pooler.is_trainable, f"Pooler {POOLER} should be trainable"


if __name__ == "__main__":
    pytest.main([__file__])
