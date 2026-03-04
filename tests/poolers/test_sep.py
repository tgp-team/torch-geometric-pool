import pytest
import torch

from tgp.poolers import SEPPooling
from tgp.select import SelectOutput, SEPSelect
from tgp.src import PoolingOutput


def _identity_select_output(num_nodes: int) -> SelectOutput:
    node_index = torch.arange(num_nodes, dtype=torch.long)
    cluster_index = torch.arange(num_nodes, dtype=torch.long)
    return SelectOutput(
        node_index=node_index,
        num_nodes=num_nodes,
        cluster_index=cluster_index,
        num_supernodes=num_nodes,
    )


def _dummy_pooled_output(num_nodes: int, so: SelectOutput) -> PoolingOutput:
    if num_nodes > 0:
        idx = torch.arange(num_nodes, dtype=torch.long)
        edge_index = torch.stack([idx, idx], dim=0)
        edge_weight = torch.ones(num_nodes, dtype=torch.float32)
        batch = torch.zeros(num_nodes, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        batch = torch.empty((0,), dtype=torch.long)

    return PoolingOutput(
        edge_index=edge_index, edge_weight=edge_weight, batch=batch, so=so
    )


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    row = torch.arange(num_nodes - 1, dtype=torch.long)
    col = row + 1
    return torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)


def test_sep_pooling_forward(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse

    pooler = SEPPooling(
        cached=False,
        connect_red_op="sum",
        lift_red_op="mean",
        s_inv_op="transpose",
    )
    pooler.eval()

    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    assert isinstance(out.so, SelectOutput)
    assert out.x.size(0) == out.so.num_supernodes
    assert out.batch is not None


def test_sep_pooling_forward_with_precomputed_so(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    pooler = SEPPooling(cached=False)
    selector = SEPSelect()
    so = selector(
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        num_nodes=x.size(0),
    )

    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, so=so)
    assert isinstance(out, PoolingOutput)
    assert out.so is so


def test_sep_pooling_lifting_branch(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse
    pooler = SEPPooling(cached=False)

    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
    x_lifted = pooler(x=out.x, so=out.so, lifting=True)

    assert isinstance(x_lifted, torch.Tensor)
    assert x_lifted.size(0) == out.so.num_nodes
    assert x_lifted.size(1) == out.x.size(1)


def test_sep_pooling_repr_and_extra_args():
    pooler = SEPPooling(cached=True)
    rep = repr(pooler)
    assert "SEPPooling" in rep
    assert pooler.extra_repr_args() == {"cached": True}


def test_sep_pooling_multi_level_precoarsening_validation_errors():
    pooler = SEPPooling(cached=False)

    with pytest.raises(ValueError, match="'levels' must be >= 1"):
        pooler.multi_level_precoarsening(levels=0, edge_index=_chain_edge_index(3))

    with pytest.raises(ValueError, match="edge_index cannot be None"):
        pooler.multi_level_precoarsening(levels=1, edge_index=None)


def test_sep_pooling_multi_level_precoarsening_single_level_branch(monkeypatch):
    pooler = SEPPooling(cached=False)
    clear_calls = {"count": 0}

    monkeypatch.setattr(
        pooler,
        "clear_cache",
        lambda: clear_calls.__setitem__("count", clear_calls["count"] + 1),
    )
    monkeypatch.setattr(
        pooler,
        "precoarsening",
        lambda **_kwargs: _dummy_pooled_output(2, _identity_select_output(2)),
    )

    out = pooler.multi_level_precoarsening(
        levels=1, edge_index=_chain_edge_index(2), num_nodes=2
    )

    assert len(out) == 1
    assert out[0].as_data().num_nodes == 2
    assert clear_calls["count"] == 2


def test_sep_pooling_multi_level_precoarsening_single_level_without_callable_clear_cache(
    monkeypatch,
):
    pooler = SEPPooling(cached=False)

    monkeypatch.setattr(pooler, "clear_cache", None, raising=False)
    monkeypatch.setattr(
        pooler,
        "precoarsening",
        lambda **_kwargs: _dummy_pooled_output(2, _identity_select_output(2)),
    )

    out = pooler.multi_level_precoarsening(
        levels=1, edge_index=_chain_edge_index(2), num_nodes=2
    )

    assert len(out) == 1
    assert out[0].as_data().num_nodes == 2


def test_sep_pooling_multi_level_precoarsening_len_mismatch(monkeypatch):
    pooler = SEPPooling(cached=False)

    monkeypatch.setattr(
        pooler.selector,
        "multi_level_select",
        lambda **_kwargs: [_identity_select_output(3)],
    )

    with pytest.raises(RuntimeError, match="returned 1 levels, expected 2"):
        pooler.multi_level_precoarsening(
            levels=2, edge_index=_chain_edge_index(3), num_nodes=3
        )


def test_sep_pooling_multi_level_precoarsening_inconsistent_sizes(monkeypatch):
    pooler = SEPPooling(cached=False)
    so_levels = [_identity_select_output(3), _identity_select_output(3)]

    monkeypatch.setattr(
        pooler.selector, "multi_level_select", lambda **_kwargs: so_levels
    )
    monkeypatch.setattr(
        pooler,
        "_precoarsening_from_select_output",
        lambda so, edge_index, edge_weight, batch=None, **_kwargs: _dummy_pooled_output(
            2, so
        ),
    )

    with pytest.raises(RuntimeError, match="Inconsistent hierarchy sizes"):
        pooler.multi_level_precoarsening(
            levels=2, edge_index=_chain_edge_index(3), num_nodes=3
        )


def test_sep_pooling_multi_level_precoarsening_success(monkeypatch):
    pooler = SEPPooling(cached=False)
    so_levels = [_identity_select_output(3), _identity_select_output(2)]
    clear_calls = {"count": 0}
    next_sizes = iter([2, 1])

    monkeypatch.setattr(
        pooler,
        "clear_cache",
        lambda: clear_calls.__setitem__("count", clear_calls["count"] + 1),
    )
    monkeypatch.setattr(
        pooler.selector, "multi_level_select", lambda **_kwargs: so_levels
    )
    monkeypatch.setattr(
        pooler,
        "_precoarsening_from_select_output",
        lambda so, edge_index, edge_weight, batch=None, **_kwargs: _dummy_pooled_output(
            next(next_sizes), so
        ),
    )

    out = pooler.multi_level_precoarsening(
        levels=2, edge_index=_chain_edge_index(3), num_nodes=3
    )

    assert len(out) == 2
    assert out[0].as_data().num_nodes == 2
    assert out[1].as_data().num_nodes == 1
    assert clear_calls["count"] == 2


def test_sep_pooling_multi_level_precoarsening_without_callable_clear_cache(
    monkeypatch,
):
    pooler = SEPPooling(cached=False)
    so_levels = [_identity_select_output(3), _identity_select_output(2)]
    next_sizes = iter([2, 1])

    monkeypatch.setattr(pooler, "clear_cache", None, raising=False)
    monkeypatch.setattr(
        pooler.selector, "multi_level_select", lambda **_kwargs: so_levels
    )
    monkeypatch.setattr(
        pooler,
        "_precoarsening_from_select_output",
        lambda so, edge_index, edge_weight, batch=None, **_kwargs: _dummy_pooled_output(
            next(next_sizes), so
        ),
    )

    out = pooler.multi_level_precoarsening(
        levels=2, edge_index=_chain_edge_index(3), num_nodes=3
    )

    assert len(out) == 2
    assert out[0].as_data().num_nodes == 2
    assert out[1].as_data().num_nodes == 1


if __name__ == "__main__":
    pytest.main([__file__])
