"""Shared test: batched vs unbatched loss equality and PoolingOutput for all dense poolers."""

import pytest
import torch
from torch_geometric.utils import to_dense_adj

from tests.test_utils import _dense_batched_to_sparse_unbatched
from tgp.poolers import (
    AsymCheegerCutPooling,
    DiffPool,
    DMoNPooling,
    HOSCPooling,
    JustBalancePooling,
    MinCutPooling,
    get_pooler,
)

# Dense poolers: those with compute_sparse_loss and batched=True support.
# JustBalance has compute_sparse_loss(S, batch); others have (edge_index, edge_weight, S, batch).
DENSE_POOLER_CONFIGS = [
    ("mincut", MinCutPooling, True),
    ("acc", AsymCheegerCutPooling, True),
    ("diff", DiffPool, True),
    ("dmon", DMoNPooling, True),
    ("hosc", HOSCPooling, True),
    ("jb", JustBalancePooling, False),
]


# Pooler names used for batched vs unbatched PoolingOutput comparison.
# Only poolers that support batched dense and unbatched sparse with the same
# (B, N, K) / (N, K) semantics (mincut, acc, diff, dmon, hosc, jb). Other
# dense poolers (eigen, lap, nmf, bnpool) have different outputs or signatures.
POOLING_OUTPUT_POOLER_NAMES = [c[0] for c in DENSE_POOLER_CONFIGS]


@pytest.mark.parametrize(
    "pooler_name,pooler_cls,uses_adj_in_sparse_loss", DENSE_POOLER_CONFIGS
)
def test_dense_pooler_batched_vs_unbatched_loss_equality(
    pooler_test_graph_dense_batch,
    pooler_name,
    pooler_cls,
    uses_adj_in_sparse_loss,
):
    """For each dense pooler, batched forward loss dict matches compute_sparse_loss."""
    x, adj = pooler_test_graph_dense_batch
    n_features = x.shape[-1]
    kwargs = {"in_channels": n_features, "k": 3, "batched": True}
    pooler = pooler_cls(**kwargs)
    pooler.eval()

    out = pooler(x=x, adj=adj)
    S = out.so.s
    edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(adj, S)

    if uses_adj_in_sparse_loss:
        loss_sparse = pooler.compute_sparse_loss(edge_index, edge_weight, S_flat, batch)
    else:
        loss_sparse = pooler.compute_sparse_loss(S_flat, batch)

    for key in out.loss:
        assert key in loss_sparse, (
            f"{pooler_name}: batched loss key {key!r} missing in sparse loss"
        )
        assert torch.allclose(out.loss[key], loss_sparse[key], rtol=1e-5, atol=1e-5), (
            f"{pooler_name}: loss key {key!r} dense={out.loss[key].item()} "
            f"vs sparse={loss_sparse[key].item()}"
        )
    for key in loss_sparse:
        assert key in out.loss, (
            f"{pooler_name}: sparse loss key {key!r} missing in batched loss"
        )


@pytest.mark.parametrize(
    "pooler_name",
    POOLING_OUTPUT_POOLER_NAMES,
    ids=POOLING_OUTPUT_POOLER_NAMES,
)
def test_dense_pooler_batched_vs_unbatched_pooling_output(
    pooler_test_graph_dense_batch,
    pooler_name,
):
    """For each dense pooler, batched and unbatched PoolingOutput coincide after reshaping."""
    x, adj = pooler_test_graph_dense_batch
    B, N, F = x.shape
    K = 3

    pooler_batched = get_pooler(
        pooler_name,
        in_channels=F,
        k=K,
        batched=True,
    )
    pooler_batched.eval()

    out_b = pooler_batched(x=x, adj=adj)

    pooler_unbatched = get_pooler(
        pooler_name + "_u",
        in_channels=F,
        k=K,
    )
    pooler_unbatched.load_state_dict(pooler_batched.state_dict())
    pooler_unbatched.eval()

    edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
        adj, out_b.so.s
    )
    x_flat = x.reshape(B * N, F)

    out_u = pooler_unbatched(
        x=x_flat,
        adj=edge_index,
        edge_weight=edge_weight,
        batch=batch,
    )

    # so.s: batched (B, N, K) -> (B*N, K); unbatched (B*N, K)
    s_b = out_b.so.s.reshape(B * N, K)
    s_u = out_u.so.s
    assert torch.allclose(s_b, s_u, rtol=1e-5, atol=1e-5), (
        f"{pooler_name}: batched and unbatched so.s should match"
    )

    # x (pooled): batched (B, K, F) -> (B*K, F); unbatched may be (B*K, F) or (B, K, F)
    x_b = out_b.x.reshape(B * K, -1)
    x_u = out_u.x
    if x_u.dim() == 3:
        x_u = x_u.reshape(B * K, -1)
    assert x_b.shape == x_u.shape, (
        f"{pooler_name}: batched and unbatched pooled x shapes must match: "
        f"batched {x_b.shape} vs unbatched {x_u.shape}. "
        "Ensure both paths use the same feature dimension (input x has F features)."
    )
    assert torch.allclose(x_b, x_u, rtol=1e-5, atol=1e-5), (
        f"{pooler_name}: batched and unbatched pooled x should match"
    )

    # edge_index / adj: batched is dense (B, K, K); unbatched may be dense or sparse
    adj_b = out_b.edge_index
    adj_u_raw = out_u.edge_index
    if adj_u_raw.dim() == 3:
        # Unbatched path already returned a dense [B, K, K] adjacency
        adj_u = adj_u_raw
    else:
        # Unbatched path returned sparse connectivity -> convert to dense
        adj_u = to_dense_adj(
            adj_u_raw,
            edge_attr=out_u.edge_weight,
            batch=out_u.batch,
            batch_size=B,
            max_num_nodes=K,
        )
    assert adj_b.shape == adj_u.shape, (
        f"{pooler_name}: batched adj {adj_b.shape} vs unbatched adj {adj_u.shape}"
    )
    assert torch.allclose(adj_b, adj_u, rtol=1e-5, atol=1e-5), (
        f"{pooler_name}: batched and unbatched pooled adj should match"
    )

    # loss: same keys and values
    assert set(out_b.loss.keys()) == set(out_u.loss.keys()), (
        f"{pooler_name}: batched and unbatched loss keys should match"
    )
    for key in out_b.loss:
        assert torch.allclose(out_b.loss[key], out_u.loss[key], rtol=1e-5, atol=1e-5), (
            f"{pooler_name}: loss key {key!r} batched={out_b.loss[key].item()} "
            f"vs unbatched={out_u.loss[key].item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
