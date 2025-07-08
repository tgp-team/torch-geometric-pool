import pytest
import torch
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

from tgp.select.ndp_select import NDPSelect


def make_chain_edge_index(N=3):
    # Build a 3-node chain 0–1–2 (undirected) without self-loops
    row = torch.tensor([0, 1, 1], dtype=torch.long)
    col = torch.tensor([1, 0, 2], dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def test_forward_no_batch_and_tensor_edge_index():
    edge_index = make_chain_edge_index(N=3)
    # Calling without edge_weight or batch
    selector = NDPSelect(s_inv_op="transpose")
    out = selector(edge_index=edge_index, edge_weight=None, batch=None, num_nodes=None)
    # Ensure SelectOutput has 's', 's_inv', 'L'
    assert hasattr(out, "s") and hasattr(out, "s_inv") and hasattr(out, "L")
    # s should be a SparseTensor of shape [3, k]
    s = out.s
    assert s.sparse_sizes()[0] == 3


def test_forward_with_sparse_tensor_edge_index_and_skip_empty_subgraph():
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
    assert s.sparse_sizes()[0] == 3
    assert len(repr(out)) > 0


def test_random_cut():
    adj = SparseTensor.from_dense(torch.ones(5, 5))
    selector = NDPSelect(s_inv_op="transpose")
    batch = torch.ones(5, dtype=torch.long)
    out = selector(edge_index=adj, batch=batch, num_nodes=5)

    # The number of clusters should be reasonable for a 5-node fully connected graph
    # Allow for some variance due to randomness in the algorithm
    assert out.num_supernodes <= 5  # At most as many clusters as nodes
    assert out.num_supernodes >= 1  # At least one cluster


if __name__ == "__main__":
    pytest.main([__file__])
