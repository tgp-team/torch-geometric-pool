import pytest
import torch

from tests.test_utils import make_chain_graph_sparse
from tgp.poolers import MinCutPooling
from tgp.select import SelectOutput
from tgp.src import (
    BasePrecoarseningMixin,
    DenseSRCPooling,
    PoolingOutput,
    Precoarsenable,
    SRCPooling,
)


class _DummyReducer(torch.nn.Module):
    @staticmethod
    def reduce_batch(select_output: SelectOutput, batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            select_output.num_supernodes, dtype=batch.dtype, device=batch.device
        )

    def reset_parameters(self):
        pass


class _DummyConnector(torch.nn.Module):
    def forward(self, so, edge_index, edge_weight=None, **kwargs):
        return edge_index, edge_weight

    def reset_parameters(self):
        pass


class _DummyPrecoarseningPooler(BasePrecoarseningMixin, SRCPooling):
    def __init__(self):
        super().__init__(
            selector=None,
            reducer=_DummyReducer(),
            lifter=None,
            connector=_DummyConnector(),
        )


class _SimplePrecoarsenable(Precoarsenable):
    def __init__(self):
        self.calls = 0

    def precoarsening(
        self,
        edge_index=None,
        edge_weight=None,
        *,
        batch=None,
        num_nodes=None,
        **kwargs,
    ) -> PoolingOutput:
        self.calls += 1

        if edge_index is None:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
        if batch is None:
            n_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 1
            batch = torch.zeros(n_nodes, dtype=torch.long)

        so = SelectOutput(
            cluster_index=torch.zeros(batch.numel(), dtype=torch.long),
            num_supernodes=1,
        )
        return PoolingOutput(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            so=so,
        )


def test_compute_loss_none():
    pooler = SRCPooling()
    assert pooler.compute_loss() is None


def test_pooling_output_get_loss_value_and_as_data_branches():
    loss = {"quality": torch.tensor(1.0), "kl": torch.tensor(2.0)}
    out = PoolingOutput(loss=loss)
    assert out.get_loss_value() == [loss["quality"], loss["kl"]]
    assert out.get_loss_value("quality") is loss["quality"]

    out_x = PoolingOutput(x=torch.randn(3, 2))
    assert out_x.as_data().num_nodes == 3

    so = SelectOutput(s=torch.randn(4, 5))
    out_so = PoolingOutput(so=so)
    assert out_so.as_data().num_nodes == 5


def test_pooling_output_as_data_with_no_fields_sets_num_nodes_none():
    out = PoolingOutput()
    data = out.as_data()
    assert data.num_nodes is None


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


def test_dense_src_preprocessing_cache_read_and_write():
    x, edge_index, edge_weight, batch = make_chain_graph_sparse(N=4, F_dim=3)
    pooler = DenseSRCPooling(cache_preprocessing=True)

    _, adj_first, _ = pooler.preprocessing(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        use_cache=True,
    )
    assert pooler.preprocessing_cache is not None

    cached_adj = torch.full_like(adj_first, 7.0)
    pooler.preprocessing_cache = cached_adj
    _, adj_second, _ = pooler.preprocessing(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        use_cache=True,
    )
    torch.testing.assert_close(adj_second, cached_adj)


def test_ensure_batched_inputs_edge_cases_and_cache_flags():
    pooler = DenseSRCPooling(cache_preprocessing=True)

    with pytest.raises(ValueError, match="edge_index cannot be None"):
        pooler._ensure_batched_inputs(
            x=torch.randn(2, 3),
            edge_index=None,
            edge_weight=None,
            batch=None,
            mask=None,
        )

    dense_adj = torch.eye(2).unsqueeze(0)

    # Explicit use_cache=False covers the non-None branch for use_cache.
    x_out, _, _ = pooler._ensure_batched_inputs(
        x=torch.randn(2, 3),
        edge_index=dense_adj,
        edge_weight=None,
        batch=None,
        mask=None,
        use_cache=False,
    )
    assert x_out.shape == (1, 2, 3)
    assert pooler.preprocessing_cache is None

    # Multi-graph batches disable cache even if use_cache=True.
    pooler_multi = DenseSRCPooling(cache_preprocessing=True)
    pooler_multi._ensure_batched_inputs(
        x=torch.randn(2, 2, 3),
        edge_index=dense_adj.repeat(2, 1, 1),
        edge_weight=None,
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        mask=None,
        use_cache=True,
    )
    assert pooler_multi.preprocessing_cache is None

    # Single-graph dense path stores preprocessing cache when requested.
    pooler_single = DenseSRCPooling(cache_preprocessing=True)
    _, adj_single, _ = pooler_single._ensure_batched_inputs(
        x=torch.randn(1, 2, 3),
        edge_index=dense_adj,
        edge_weight=None,
        batch=torch.zeros(2, dtype=torch.long),
        mask=None,
        use_cache=True,
    )
    assert pooler_single.preprocessing_cache is not None
    torch.testing.assert_close(pooler_single.preprocessing_cache, adj_single)


def test_finalize_sparse_output_covers_batch_generation_paths():
    pooler = MinCutPooling(in_channels=2, k=3, sparse_output=True)

    # batch_pooled inferred via reducer.reduce_batch when batch is provided.
    so_dense_two_graphs = SelectOutput(
        s=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.float,
        )
    )
    x_pool = torch.randn(2, 2, 2)
    adj_pool = torch.eye(2).repeat(2, 1, 1)
    x_out, edge_index, edge_weight, batch_pooled = pooler._finalize_sparse_output(
        x_pool=x_pool,
        adj_pool=adj_pool,
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        batch_pooled=None,
        so=so_dense_two_graphs,
    )
    assert batch_pooled is not None
    assert batch_pooled.numel() == x_out.size(0)
    assert edge_index.size(0) == 2
    assert edge_weight.numel() == edge_index.size(1)

    # Single-graph dense path with mask and no batch builds zero batch_pooled.
    so_dense_single_graph = SelectOutput(
        s=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float)
    )
    x_out, _, _, batch_pooled = pooler._finalize_sparse_output(
        x_pool=torch.randn(1, 3, 2),
        adj_pool=torch.eye(3).unsqueeze(0),
        batch=None,
        batch_pooled=None,
        so=so_dense_single_graph,
    )
    assert batch_pooled is not None
    assert torch.all(batch_pooled == 0)
    assert batch_pooled.numel() == x_out.size(0)

    # Sparse assignments have out_mask=None and go through the dense_to_block_diag fallback.
    so_sparse = SelectOutput(
        cluster_index=torch.tensor([0, 1, 0], dtype=torch.long),
        num_supernodes=2,
    )
    x_out, edge_index, edge_weight, batch_pooled = pooler._finalize_sparse_output(
        x_pool=torch.randn(1, 2, 2),
        adj_pool=torch.tensor([[[1.0, 0.5], [0.5, 1.0]]], dtype=torch.float),
        batch=None,
        batch_pooled=None,
        so=so_sparse,
    )
    assert batch_pooled is None
    assert x_out.shape == (2, 2)
    assert edge_index.shape[0] == 2
    assert edge_weight.numel() == edge_index.shape[1]


def test_multi_level_precoarsening_validation_and_no_clear_cache_branch():
    precoarsenable = _SimplePrecoarsenable()

    with pytest.raises(ValueError, match="'levels' must be >= 1"):
        precoarsenable.multi_level_precoarsening(
            levels=0,
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        )

    outputs = precoarsenable.multi_level_precoarsening(
        levels=2,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.ones(2, dtype=torch.float),
        batch=torch.zeros(2, dtype=torch.long),
        num_nodes=2,
    )
    assert len(outputs) == 2
    assert precoarsenable.calls == 2


def test_precoarsening_from_select_output_infers_batch_when_missing():
    pooler = _DummyPrecoarseningPooler()
    so = SelectOutput(
        cluster_index=torch.tensor([0, 0, 0], dtype=torch.long),
        num_supernodes=1,
    )

    out = pooler._precoarsening_from_select_output(
        so=so,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_weight=torch.ones(2, dtype=torch.float),
        batch=None,
    )

    assert so.batch is not None
    assert so.batch.tolist() == [0, 0, 0]
    assert out.batch is not None
    assert out.batch.numel() == 1
    assert int(out.batch[0].item()) == 0


def test_precoarsening_from_select_output_uses_existing_so_batch():
    pooler = _DummyPrecoarseningPooler()
    so = SelectOutput(
        cluster_index=torch.tensor([0, 0, 0], dtype=torch.long),
        num_supernodes=1,
        batch=torch.tensor([1, 1, 1], dtype=torch.long),
    )

    out = pooler._precoarsening_from_select_output(
        so=so,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_weight=torch.ones(2, dtype=torch.float),
        batch=None,
    )

    assert torch.equal(so.batch, torch.tensor([1, 1, 1], dtype=torch.long))
    assert out.batch is not None
    assert int(out.batch[0].item()) == 0


if __name__ == "__main__":
    pytest.main([__file__])
