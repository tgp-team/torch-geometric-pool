import pytest

from tests.test_utils import make_chain_graph_sparse
from tgp.poolers import MinCutPooling
from tgp.src import SRCPooling


def test_compute_loss_none():
    pooler = SRCPooling()
    assert pooler.compute_loss() is None


def test_preprocessing():
    x, edge_index, edge_weight, batch = make_chain_graph_sparse(N=4, F_dim=5)

    # add a trailing dimension to edge_weight to simulate a feature dimension
    edge_weight = edge_weight.unsqueeze(-1)

    pooler = MinCutPooling(
        k=2,
        in_channels=x.size(-1),
    )
    x, adj, mask = pooler.preprocessing(
        edge_index=edge_index, edge_weight=edge_weight, x=x, batch=batch
    )

    assert adj.dim() == 3


if __name__ == "__main__":
    pytest.main([__file__])
