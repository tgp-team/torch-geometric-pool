import pytest
import torch

from tgp.poolers import GraclusPooling
from tgp.select import SelectOutput
from tgp.src import PoolingOutput


@pytest.fixture(scope="module")
def make_chain_graph(N=4, F_dim=5):
    row = torch.arange(N - 1, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack(
        [torch.cat([row, col]), torch.cat([col, row])], dim=0
    )  # undirected chain
    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)
    return x, edge_index, edge_weight, batch


def test_graclus_forward(make_chain_graph):
    x, edge_index, edge_weight, batch = make_chain_graph

    pooler = GraclusPooling(reduce_red_op="any", s_inv_op="inverse")
    pooler.eval()

    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    assert len(repr(out)) > 0
    assert isinstance(next(iter(out)), torch.Tensor)
    assert out.has_loss is False
    assert out.get_loss_value() == 0.0
    assert isinstance(pooler.global_pool(x, batch), torch.Tensor)
    assert pooler.get_forward_signature() is not None
    assert pooler.data_transforms() is None


def test_caching(make_chain_graph):
    x, edge_index, edge_weight, batch = make_chain_graph

    pooler = GraclusPooling(reduce_red_op="any", cached=True)
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


if __name__ == "__main__":
    pytest.main([__file__])
