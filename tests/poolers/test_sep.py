import pytest
import torch

from tgp.poolers import SEPPooling
from tgp.select import SelectOutput, SEPSelect
from tgp.src import PoolingOutput


def test_sep_pooling_forward(pooler_test_graph_sparse):
    x, edge_index, edge_weight, batch = pooler_test_graph_sparse

    pooler = SEPPooling(
        cached=False,
        reduce_red_op="mean",
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


if __name__ == "__main__":
    pytest.main([__file__])
