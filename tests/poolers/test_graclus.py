import pytest
import torch

from tgp.poolers import GraclusPooling
from tgp.reduce import readout
from tgp.select import SelectOutput
from tgp.src import PoolingOutput


def test_graclus_forward(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse

    pooler = GraclusPooling(s_inv_op="inverse")
    pooler.eval()

    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    assert len(repr(out)) > 0
    assert isinstance(next(iter(out)), torch.Tensor)
    assert out.has_loss is False
    assert out.get_loss_value() == 0.0
    assert isinstance(readout(x, batch=batch), torch.Tensor)
    assert pooler.get_forward_signature() is not None
    assert pooler.data_transforms() is None


def test_caching(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse

    pooler = GraclusPooling(cached=True)
    pooler.eval()

    # First forward pass should cache the results
    out1 = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert pooler.cached is True

    # Second forward pass should use cached results
    out2 = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(pooler._so_cached, SelectOutput)
    assert out1.x.equal(out2.x)  # Ensure the output features are the same


def test_graclus_edge_weights(pooler_test_graph_sparse):
    x, edge_index, _, batch = pooler_test_graph_sparse

    E = edge_index.size(1)
    edge_weight = torch.ones((E, 1), dtype=torch.float)

    # Test with no edge weights
    pooler = GraclusPooling(s_inv_op="inverse")
    pooler.eval()

    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    assert out.edge_index is not None
    assert out.edge_weight is not None  # No edge weights should be returned

    # Check illegal edge weights not [E] or [E, 1]
    edge_weight = torch.ones((E, 2), dtype=torch.float)

    # check that it raises an error
    with pytest.raises(RuntimeError):
        out = pooler(
            x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
        )


if __name__ == "__main__":
    pytest.main([__file__])
