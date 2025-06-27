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


def test_assign_all_nodes_weight_size_mismatch():
    """Test that assign_all_nodes raises ValueError when weight size doesn't match num_nodes."""
    # Create a SelectOutput with 4 nodes, but only 2 selected
    cluster_index = torch.tensor([0, 1])  # 2 selected nodes
    node_index = torch.tensor([0, 2])     # indices of selected nodes
    so = SelectOutput(
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=4,  # total 4 nodes
        num_clusters=2
    )
    
    # Create edge connectivity for 4 nodes
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    
    # Create weight tensor with wrong size (3 instead of 4)
    wrong_weight = torch.tensor([1.0, 2.0, 3.0])  # size 3, should be 4
    
    # This should raise ValueError due to size mismatch
    with pytest.raises(ValueError, match="Weight tensor size \\(3\\) must match the number of nodes \\(4\\)"):
        so.assign_all_nodes(adj=edge_index, weight=wrong_weight, strategy="closest_node")


def test_assign_all_nodes_with_extra_args():
    """Test that assign_all_nodes copies extra attributes from the original SelectOutput."""
    # Create a SelectOutput with some extra arguments
    cluster_index = torch.tensor([0, 1])
    node_index = torch.tensor([0, 2])
    so = SelectOutput(
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=4,
        num_clusters=2,
        custom_attr="test_value",  # extra attribute
        another_attr=42           # another extra attribute
    )
    
    # Verify the extra args are stored
    assert hasattr(so, '_extra_args')
    assert 'custom_attr' in so._extra_args
    assert 'another_attr' in so._extra_args
    assert so.custom_attr == "test_value"
    assert so.another_attr == 42
    
    # Create edge connectivity
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    
    # Call assign_all_nodes which should copy extra attributes
    new_so = so.assign_all_nodes(adj=edge_index, strategy="closest_node")
    
    # Verify that the extra attributes were copied to the new SelectOutput
    assert hasattr(new_so, 'custom_attr')
    assert hasattr(new_so, 'another_attr')
    assert new_so.custom_attr == "test_value"
    assert new_so.another_attr == 42


def test_assign_all_nodes_no_extra_args():
    """Test that assign_all_nodes works correctly when there are no extra args."""
    # Create a SelectOutput without extra arguments
    cluster_index = torch.tensor([0, 1])
    node_index = torch.tensor([0, 2])
    so = SelectOutput(
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=4,
        num_clusters=2
    )
    
    # Ensure no extra args
    assert not hasattr(so, '_extra_args') or len(so._extra_args) == 0
    
    # Create edge connectivity
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    
    # This should work without errors even with no extra args
    new_so = so.assign_all_nodes(adj=edge_index, strategy="closest_node")
    
    # Verify the result is valid
    assert isinstance(new_so, SelectOutput)
    assert new_so.num_nodes == 4
    assert new_so.num_clusters == 2


def test_assign_all_nodes_with_sparse_tensor_adj():
    """Test assign_all_nodes with SparseTensor adjacency to cover line 291."""
    # Create a SelectOutput with some nodes not selected
    cluster_index = torch.tensor([0, 1])  # 2 selected nodes
    node_index = torch.tensor([0, 2])     # indices of selected nodes
    so = SelectOutput(
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=4,
        num_clusters=2
    )
    
    # Create SparseTensor adjacency matrix to trigger line 291
    row = torch.tensor([0, 1, 2, 3])
    col = torch.tensor([1, 0, 3, 2])
    sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    
    # This should trigger the SparseTensor branch at line 291
    new_so = so.assign_all_nodes(adj=sparse_adj, strategy="closest_node")
    
    # Verify the result is valid
    assert isinstance(new_so, SelectOutput)
    assert new_so.num_nodes == 4
    assert new_so.num_clusters == 2


def test_assign_all_nodes_extra_attr_exists():
    """Test that assign_all_nodes properly handles case where extra attribute exists on original object."""
    # Create a SelectOutput with extra arguments that will exist
    cluster_index = torch.tensor([0, 1])
    node_index = torch.tensor([0, 2])
    so = SelectOutput(
        cluster_index=cluster_index,
        node_index=node_index,
        num_nodes=4,
        num_clusters=2,
        existing_attr="test_value"  # This attribute exists
    )
    
    # Manually add an attribute name to _extra_args that doesn't actually exist on the object
    # This will test the hasattr check on line 325
    so._extra_args.add('non_existent_attr')
    
    # Create edge connectivity
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    
    # Call assign_all_nodes - this should handle the case where attr_name is in _extra_args
    # but the attribute doesn't actually exist (covering the hasattr check)
    new_so = so.assign_all_nodes(adj=edge_index, strategy="closest_node")
    
    # Verify that existing_attr was copied (line 326: setattr call)
    assert hasattr(new_so, 'existing_attr')
    assert new_so.existing_attr == "test_value"
    
    # Verify that non_existent_attr was not copied (due to hasattr check on line 325)
    assert not hasattr(new_so, 'non_existent_attr')


if __name__ == "__main__":
    pytest.main([__file__])
