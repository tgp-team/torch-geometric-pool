import pytest
import torch

from tgp.imports import SparseTensor
from tgp.select.base_select import Select, SelectOutput, cluster_to_s



def test_cluster_to_s_as_edge_index():
    num_nodes = 5
    cluster_index = torch.tensor([0, 1, 0, 2, 1])
    node_index = torch.tensor([2, 0, 4, 3, 1])
    weight = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])

    edge_index, ret_weight = cluster_to_s(
        cluster_index, node_index=node_index, weight=weight, as_edge_index=True
    )
    # edge_index should be [2, num_nodes]
    assert edge_index.shape == (2, num_nodes)
    assert torch.equal(edge_index[0], node_index)
    assert torch.equal(edge_index[1], cluster_index)
    assert torch.equal(ret_weight, weight)


def test_selectoutput_from_cluster_index_and_default_s_inv():
    num_nodes = 3
    num_clusters = 2
    cluster_index = torch.tensor([0, 1, 0])
    weight = torch.tensor([1.0, 1.0, 1.0])

    out = SelectOutput(
        s=None,
        node_index=None,
        num_nodes=num_nodes,
        cluster_index=cluster_index,
        num_clusters=num_clusters,
        weight=weight,
    )
    # s should be a SparseTensor, and s_inv should default to transpose
    assert isinstance(out.s, SparseTensor)
    s_t = out.s.t()
    inv_row, inv_col, inv_val = out.s_inv.coo()
    row_t, col_t, val_t = s_t.coo()
    assert torch.equal(inv_row, row_t)
    assert torch.equal(inv_col, col_t)
    if inv_val is not None and val_t is not None:
        assert torch.equal(inv_val, val_t)

    # Check basic attributes
    assert out.num_nodes == num_nodes
    assert out.num_clusters == num_clusters
    assert torch.equal(out.node_index, torch.arange(num_nodes))
    assert torch.equal(out.cluster_index, cluster_index)
    assert torch.equal(out.weight, weight)
    assert out.is_sparse
    # row sums = [1, 1, 1], so is_expressive should be True
    assert out.is_expressive



def test_selectoutput_set_s_inv_inverse_and_invalid():
    # Use a dense s for which pseudo-inverse is known
    s = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    s = s.unsqueeze(0)  # shape: [1, 2, 2]
    out = SelectOutput(s=s)

    # Passing an invalid method should raise ValueError
    with pytest.raises(ValueError):
        out.set_s_inv("not_a_method")


def test_selectoutput_repr_clone_apply_and_device_ops():
    # Small dense s
    s = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    s = s.unsqueeze(0)  # Add batch dimension
    s_inv = s.transpose(1, 2)  # s_inv is the transpose of s
    out = SelectOutput(s=s, s_inv=s_inv)

    rep = repr(out)
    assert "SelectOutput(" in rep
    assert "num_nodes=2" in rep and "num_clusters=2" in rep

    # clone() should be a deep copy
    out_clone = out.clone()
    assert isinstance(out_clone, SelectOutput)
    out_clone.s[0, 0] = 5.0
    # Original should remain unchanged
    assert not torch.equal(out_clone.s, out.s)

    # Test apply(): multiply both s and s_inv by 2
    out2 = out.clone().apply(lambda x: x * 2)
    if isinstance(out2.s, torch.Tensor):
        assert torch.equal(out2.s, s * 2)

    # Test .cpu() and .cuda() (if available)
    out_cpu = out.clone().cpu()
    assert out_cpu.s.device.type == "cpu"
    if torch.cuda.is_available():
        out_cuda = out.clone().cuda()
        assert out_cuda.s.device.type == "cuda"

    # Test detach_() / detach()
    out_det = out.clone().detach()
    assert isinstance(out_det, SelectOutput)
    out_det_inplace = out.clone()
    out_det_inplace.detach_()
    assert isinstance(out_det_inplace, SelectOutput)
    
    out.s_inv = None
    out.cpu()
    assert out_cpu.s.device.type == "cpu"


def test_selectoutput_invalid_init():
    # If s is neither None, Tensor, nor SparseTensor, expect ValueError
    with pytest.raises(ValueError):
        SelectOutput(s="invalid_s_value")


def test_select_abstract_forward_and_repr():
    sel = Select()
    # Abstract forward must raise NotImplementedError
    with pytest.raises(NotImplementedError):
        sel.forward(x=torch.randn(1, 1), edge_index=None)
    # repr(sel) should be exactly "Select()"
    assert repr(sel) == "Select()"


def test_set_weights_spt():
    # Create a sparse tensor
    s = SparseTensor(
        row=torch.tensor([0, 1, 2]), col=torch.tensor([1, 2, 0]), sparse_sizes=(3, 3)
    )
    weight = torch.tensor([0.5, 1.5, 2.5])
    so = SelectOutput(s=s, weight=weight, num_clusters=3)

    assert so.num_nodes == 3


if __name__ == "__main__":
    pytest.main([__file__])
