import pytest
import torch
from torch_geometric.utils import add_self_loops

from tests.test_utils import make_chain_edge_index
from tgp.select.ndp_select import NDPSelect


def test_forward_no_batch_and_tensor_edge_index():
    edge_index = make_chain_edge_index(N=3)
    # Calling without edge_weight or batch
    selector = NDPSelect(s_inv_op="transpose")
    out = selector(edge_index=edge_index, edge_weight=None, batch=None, num_nodes=None)
    # Ensure SelectOutput has 's', 's_inv', 'L'
    assert hasattr(out, "s") and hasattr(out, "s_inv") and hasattr(out, "L")
    # s should be a torch COO sparse tensor of shape [3, k]
    s = out.s
    assert s.size(0) == 3


@pytest.mark.torch_sparse
def test_forward_with_sparse_tensor_edge_index_and_skip_empty_subgraph():
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    edge_index = make_chain_edge_index(N=3)
    edge_index, _ = add_self_loops(edge_index, num_nodes=3)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    spt = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)

    # Create a batch of size 3 where class '1' is empty:
    batch = torch.tensor([0, 2, 2], dtype=torch.long)
    selector = NDPSelect(s_inv_op="transpose")
    out = selector(edge_index=spt, edge_weight=edge_weight, batch=batch, num_nodes=3)
    # s should cover nodes from subgraphs 0 and 2; shape rows = 3
    s = out.s
    assert s.size(0) == 3
    assert len(repr(out)) > 0


@pytest.mark.torch_sparse
def test_random_cut():
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    n = 5
    adj = SparseTensor.from_dense(torch.ones(n, n))
    selector = NDPSelect(s_inv_op="transpose")
    batch = torch.ones(n, dtype=torch.long)
    out = selector(edge_index=adj, batch=batch, num_nodes=n)

    assert out.num_supernodes < n  # Not all nodes should be selected
    assert out.num_supernodes >= 1  # At least one node must be selected

    n = 1
    adj = SparseTensor.from_dense(torch.ones(n, n))
    selector = NDPSelect(s_inv_op="transpose")
    batch = torch.ones(n, dtype=torch.long)
    out = selector(edge_index=adj, batch=batch, num_nodes=n)

    assert out.num_supernodes == 1


if __name__ == "__main__":
    pytest.main([__file__])
