import pytest
import torch
from torch_geometric.nn import DenseGCNConv, GCNConv

from tgp.poolers import get_pooler, pooler_map
from tgp.select.base_select import SelectOutput

poolers = list(pooler_map.keys())
excluded_poolers = ["pan"]
poolers = [p for p in poolers if p not in excluded_poolers]


@pytest.mark.parametrize("pooler_name", poolers)
def test_poolers_forward_and_lifting(pooler_test_graph_sparse_batch_tuple, pooler_name):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse_batch_tuple
    N, F = x.size()

    # Common parameters for all poolers (some may ignore irrelevant keys):
    PARAMS = {
        "in_channels": F,
        "ratio": 0.5,
        "k": max(1, N // 2),
        "cached": True,
        "lift": "precomputed",
        "s_inv_op": "transpose",
        "lift_red_op": "mean",
        "loss_coeff": 1.0,
        "remove_self_loops": True,
        "scorer": "degree",
        "reduce": "sum",
    }

    # Instantiate the pooler
    pooler = get_pooler(pooler_name, **PARAMS)
    pooler.eval()
    print(f"Testing pooler: {pooler_name}")

    # Forward pass pooling
    out = pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
    assert hasattr(out, "x")
    assert hasattr(out, "so") and isinstance(out.so, SelectOutput)
    assert isinstance(out.x, torch.Tensor)
    num_supernodes = out.so.num_supernodes
    # For batched graphs, some dense poolers create k supernodes per graph,
    # so num_supernodes can be batch_size * k, which may exceed N.
    # We check that we have at least 1 supernode and at most batch_size * k.
    if batch is not None:
        batch_size = int(batch.max().item()) + 1
        k = PARAMS.get("k", max(1, N // 2))
        max_expected_supernodes = batch_size * k
        assert 1 <= num_supernodes <= max_expected_supernodes
    else:
        assert 1 <= num_supernodes <= N

    # For dense [N, K] assignment matrices with multi-graph batches (e.g., SparseBNPool),
    # the actual output has B*K nodes, while num_supernodes = K (clusters per graph).
    # For sparse block-diagonal assignments, num_supernodes = B*K.
    actual_max_supernodes = (
        max_expected_supernodes if batch is not None else num_supernodes
    )
    assert 1 <= out.x.size(0) <= actual_max_supernodes

    ei = out.edge_index
    assert isinstance(ei, torch.Tensor)

    # If edge_weight present, check shape consistency
    if hasattr(out, "edge_weight") and out.edge_weight is not None:
        ew = out.edge_weight
        assert isinstance(ew, torch.Tensor)
        assert ew.numel() == ei.size(1)

    # Apply message passing to ensure output is correct type
    # Skip message passing for EigenPooling as its output structure (adjacency [K, K],
    # features [B*K, H*F]) doesn't match standard dense or sparse GCN expectations.
    if pooler_name != "eigen":
        out_features = out.x.size(-1)
        use_dense_mp = pooler.is_dense and not getattr(pooler, "sparse_output", False)
        conv = (
            DenseGCNConv(out_features, out_features)
            if use_dense_mp
            else GCNConv(out_features, out_features)
        )
        out.x = conv(out.x, out.edge_index)
        assert isinstance(out.x, torch.Tensor)

    # Lifting check
    # Skip lifting test for EigenPooling in this generic test - eigenpool's complex
    # reshape behavior (H*d <-> d) with multi-graph batches is thoroughly tested in
    # tests/poolers/test_eigenpool.py and tests/lift/test_eigenpool_lift.py.
    if pooler_name != "eigen":
        x_pool = out.x.clone()
        lift_kwargs = {}
        if pooler_name in {"lap", "spbnpool"}:
            lift_kwargs["batch_pooled"] = out.batch
        x_lifted = pooler(x=x_pool, so=out.so, lifting=True, **lift_kwargs)
        assert isinstance(x_lifted, torch.Tensor)
        # For batched graphs, the lifted features should match the number of nodes
        # that the SelectOutput knows about. Some poolers may only lift one graph at a time
        # when dealing with batched graphs, so we check against out.so.num_nodes instead of N.
        assert x_lifted.size(-2) == out.so.num_nodes
        assert x_lifted.size(-1) == x_pool.size(-1)

    # Aux loss check
    if hasattr(out, "loss") and out.loss is not None:
        for loss_val in out.loss.keys():
            # assert isinstance(loss_val, torch.Tensor) or isinstance(loss_val, float)
            assert isinstance(out.loss[loss_val], (torch.Tensor, float)), (
                f"Loss value {loss_val} should be a tensor or float, got {type(out.loss[loss_val])}"
            )

    # repr check
    rep = repr(pooler)
    assert pooler_map[pooler_name].__name__.lower() in rep.lower()

    # Cache check
    if hasattr(pooler, "clear_cache"):
        pooler.clear_cache()

    # reset params check
    if hasattr(pooler, "reset_parameters"):
        pooler.reset_parameters()

    out2 = pooler(x=x, adj=edge_index, batch=batch)
    assert isinstance(out2, type(out))


def test_wrong_pooler_name():
    with pytest.raises(ValueError):
        get_pooler("non_existent_pooler")


if __name__ == "__main__":
    pytest.main([__file__])
