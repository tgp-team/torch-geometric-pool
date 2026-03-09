import pytest
import torch

from tgp.poolers.sag import SAGPooling
from tgp.src import PoolingOutput


class DummyGNN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index):
        del edge_index
        return torch.ones(
            (x.size(0), self.out_channels), device=x.device, dtype=x.dtype
        )


def test_sag_signature_fallback_on_type_error(monkeypatch):
    calls = {"count": 0}

    def _raise_signature(_):
        calls["count"] += 1
        raise TypeError("signature unavailable")

    monkeypatch.setattr("tgp.poolers.sag.inspect.signature", _raise_signature)

    pooler = SAGPooling(
        in_channels=4,
        ratio=0.5,
        GNN=DummyGNN,
        ignored_kwarg="ignored",
    )

    assert calls["count"] == 1
    assert isinstance(pooler.gnn, DummyGNN)
    assert pooler.gnn.in_channels == 4
    assert pooler.gnn.out_channels == 1


def test_sag_forward_smoke(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    pooler = SAGPooling(in_channels=x.size(-1), ratio=0.5)
    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)

    assert isinstance(out, PoolingOutput)
    assert out.x.dim() == 2
    assert out.edge_index.dim() == 2


if __name__ == "__main__":
    pytest.main([__file__])
