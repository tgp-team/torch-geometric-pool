import ctypes
import inspect

import numpy as np
import pytest
import torch

from tgp.select.sep_select import (
    PartitionTree,
    PartitionTreeNode,
    SEPSelect,
    _adj_mat_to_coding_tree,
    _child_tree_depth,
    _combine_delta,
    _compress_delta,
    _compress_node,
    _cut_volume,
    _depth_one_assignment,
    _graph_parse,
    _id_generator,
    _layer_first,
    _make_select_output,
    _merge_nodes,
    _split_subgraphs,
    _trans_to_tree,
    _update_depth,
    _update_node,
)


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    row = torch.arange(num_nodes - 1, dtype=torch.long)
    col = row + 1
    return torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)


def test_sep_select_forward_basic_and_repr():
    selector = SEPSelect(s_inv_op="transpose")
    edge_index = _chain_edge_index(6)
    out = selector(edge_index=edge_index, num_nodes=6)

    assert out.num_nodes == 6
    assert out.num_supernodes >= 1
    assert out.cluster_index.numel() == 6
    assert torch.all(out.cluster_index >= 0)
    assert "SEPSelect" in repr(selector)


def test_sep_select_forward_batched_with_edge_weight_and_deterministic():
    selector = SEPSelect()

    edge_index_g0 = _chain_edge_index(3)
    edge_index_g1 = _chain_edge_index(3) + 3
    edge_index = torch.cat([edge_index_g0, edge_index_g1], dim=1)
    edge_weight = torch.arange(1, edge_index.size(1) + 1, dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    out1 = selector(
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        num_nodes=7,
    )
    out2 = selector(
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        num_nodes=7,
    )

    assert out1.num_nodes == 7
    assert out1.cluster_index.numel() == 7
    assert out1.num_supernodes >= 2
    assert torch.equal(out1.cluster_index, out2.cluster_index)


def test_sep_select_forward_infers_num_nodes_from_batch_and_skips_empty_batch_id():
    selector = SEPSelect()
    # Batch id "1" has no nodes on purpose.
    batch = torch.tensor([0, 2, 2], dtype=torch.long)
    edge_index = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)

    out = selector(edge_index=edge_index, batch=batch, num_nodes=None)
    assert out.num_nodes == 3
    assert out.cluster_index.numel() == 3
    assert out.num_supernodes >= 2


def test_sep_select_forward_empty_and_bad_batch_size():
    selector = SEPSelect()
    edge_index = torch.empty((2, 0), dtype=torch.long)

    out = selector(edge_index=edge_index, num_nodes=0)
    assert out.num_nodes == 0
    assert out.num_supernodes == 0
    assert out.cluster_index.numel() == 0

    with pytest.raises(ValueError):
        selector(
            edge_index=edge_index,
            batch=torch.tensor([0, 0], dtype=torch.long),
            num_nodes=3,
        )


def test_split_subgraphs_and_cluster_branches():
    empty_split = _split_subgraphs(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        batch=torch.empty((0,), dtype=torch.long),
        num_nodes=0,
    )
    assert empty_split == []

    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.empty((0,), dtype=torch.float)
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    subgraphs = _split_subgraphs(edge_index, edge_weight, batch, num_nodes=3)
    assert len(subgraphs) == 2

    selector = SEPSelect()
    out = selector(
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        num_nodes=3,
    )
    assert out.cluster_index.tolist() == [0, 1, 2]
    assert out.num_supernodes == 3

    # Cover branch where a graph id has no nodes.
    split_with_gap = _split_subgraphs(
        edge_index=torch.tensor([[1, 2], [2, 1]], dtype=torch.long),
        edge_weight=None,
        batch=torch.tensor([0, 2, 2], dtype=torch.long),
        num_nodes=3,
    )
    assert len(split_with_gap) == 2


def test_depth_assignment_and_tree_builders():
    tree_nodes = {
        0: {"ID": 0, "depth": 0, "graphID": 0},
        1: {"ID": 1, "depth": 0, "graphID": 1},
        2: {"ID": 2, "depth": 1, "children": [0, 1]},
    }
    assignment = _depth_one_assignment(
        tree_nodes, num_nodes=3, device=torch.device("cpu")
    )
    assert assignment.tolist() == [0, 0, 1]

    tree_nodes_with_empty = {
        0: {"ID": 0, "depth": 0, "graphID": 0},
        1: {"ID": 1, "depth": 0, "graphID": 1},
        2: {"ID": 2, "depth": 1, "children": None},
    }
    assignment = _depth_one_assignment(
        tree_nodes_with_empty,
        num_nodes=2,
        device=torch.device("cpu"),
    )
    assert assignment.tolist() == [0, 1]

    empty_tree = _adj_mat_to_coding_tree(np.zeros((0, 0)), tree_depth=2)
    assert empty_tree == {}

    # Two disconnected components.
    adj = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    tree = _adj_mat_to_coding_tree(adj, tree_depth=2)
    roots = [nid for nid, node in tree.items() if node["parent"] is None]
    assert len(roots) == 1
    assert tree[roots[0]]["depth"] == 2

    conn_adj = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )
    trans_tree = _trans_to_tree(conn_adj, tree_depth=2)
    assert len(trans_tree) >= 3


def test_low_level_helpers():
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ]
    )
    g_num_nodes, vol, node_vol, adj_table = _graph_parse(adj)
    assert g_num_nodes == 3
    assert vol == 6.0
    assert node_vol == [1.0, 3.0, 2.0]
    assert adj_table[1] == {0, 2}

    cut = _cut_volume(adj, np.array([0, 1]), np.array([2]))
    assert cut == 2.0
    assert _cut_volume(adj, np.array([], dtype=int), np.array([1])) == 0.0

    gen = _id_generator()
    assert next(gen) == 0
    assert next(gen) == 1

    n1 = PartitionTreeNode(ID=0, partition=[0], vol=2.0, g=1.0, child_cut=0.5)
    n2 = PartitionTreeNode(ID=1, partition=[1], vol=3.0, g=1.2)
    parent = PartitionTreeNode(ID=2, partition=[0, 1], vol=5.0, g=1.0)
    assert _compress_delta(n1, parent) > 0
    assert np.isfinite(_combine_delta(n1, n2, cut_v=0.4, graph_vol=7.0))


def test_tree_structural_helpers():
    node_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[0, 1],
            vol=2.0,
            g=1.0,
            children={0, 1},
            parent=3,
            child_h=1,
            child_cut=0.2,
        ),
        3: PartitionTreeNode(
            ID=3,
            partition=[0, 1, 4],
            vol=3.0,
            g=1.0,
            children={2, 4},
            parent=None,
            child_h=2,
            child_cut=0.1,
        ),
        4: PartitionTreeNode(ID=4, partition=[4], vol=1.0, g=1.0, parent=3, child_h=0),
    }

    bfs = list(_layer_first(node_dict, 3))
    assert bfs[0] == 3
    assert set(bfs) == set(node_dict.keys())

    _compress_node(node_dict, node_id=2, parent_id=3)
    assert 2 not in node_dict
    assert node_dict[3].children == {0, 1, 4}
    assert node_dict[0].parent == 3
    assert node_dict[1].parent == 3
    assert _child_tree_depth(node_dict, 0) >= 1

    # Cover branch where parent height does not need upward propagation.
    node_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[0, 1],
            vol=2.0,
            g=1.0,
            children={0, 1},
            parent=3,
            child_h=1,
            child_cut=0.2,
        ),
        3: PartitionTreeNode(
            ID=3,
            partition=[0, 1, 4],
            vol=3.0,
            g=1.0,
            children={2, 4},
            parent=None,
            child_h=4,
            child_cut=0.1,
        ),
        4: PartitionTreeNode(ID=4, partition=[4], vol=1.0, g=1.0, parent=3, child_h=0),
    }
    _compress_node(node_dict, node_id=2, parent_id=3)
    assert 2 not in node_dict

    # Cover branch that breaks immediately inside the while loop.
    node_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[0, 1],
            vol=2.0,
            g=1.0,
            children={0, 1},
            parent=None,
            child_h=1,
            child_cut=0.2,
        ),
    }
    _compress_node(node_dict, node_id=0, parent_id=2)
    assert 0 not in node_dict

    # Cover upward propagation across multiple ancestors.
    node_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            parent=2,
            child_h=0,
            children=set(),
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[0, 1],
            vol=2.0,
            g=1.0,
            children={0, 1},
            parent=3,
            child_h=1,
            child_cut=0.2,
        ),
        3: PartitionTreeNode(
            ID=3,
            partition=[0, 1, 2],
            vol=3.0,
            g=1.0,
            children={2, 4},
            parent=5,
            child_h=2,
            child_cut=0.1,
        ),
        4: PartitionTreeNode(ID=4, partition=[2], vol=1.0, g=1.0, parent=3, child_h=0),
        5: PartitionTreeNode(
            ID=5,
            partition=[0, 1, 2, 3],
            vol=4.0,
            g=1.0,
            children={3, 6},
            parent=None,
            child_h=4,
            child_cut=0.1,
        ),
        6: PartitionTreeNode(ID=6, partition=[3], vol=1.0, g=1.0, parent=5, child_h=0),
    }
    _compress_node(node_dict, node_id=2, parent_id=3)
    assert node_dict[3].parent == 5

    merge_nodes = {
        0: PartitionTreeNode(ID=0, partition=[0], vol=1.0, g=1.0, child_h=0),
        1: PartitionTreeNode(ID=1, partition=[1], vol=2.0, g=2.0, child_h=0),
    }
    _merge_nodes(new_id=2, id1=0, id2=1, cut_v=0.5, node_dict=merge_nodes)
    assert merge_nodes[0].parent == 2
    assert merge_nodes[1].parent == 2
    assert merge_nodes[2].children == {0, 1}


def test_update_depth_and_update_node():
    tree = {
        0: PartitionTreeNode(ID=0, partition=[0], vol=1.0, g=1.0, parent=2, child_h=0),
        1: PartitionTreeNode(ID=1, partition=[1], vol=1.0, g=1.0, parent=2, child_h=0),
        2: PartitionTreeNode(
            ID=2,
            partition=[0, 1],
            vol=2.0,
            g=1.0,
            parent=None,
            children={0, 1},
            child_h=1,
        ),
    }
    _update_depth(tree)
    assert tree[0].child_h == 0
    assert tree[2].child_h >= 1

    new_tree = _update_node(tree)
    assert isinstance(new_tree, dict)
    assert all("depth" in node for node in new_tree.values())


def test_partition_tree_modes_and_helpers():
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    tree = PartitionTree(adj)
    tree.build_coding_tree(k=1)

    tree = PartitionTree(adj)
    tree.build_coding_tree(k=2, mode="v1")
    assert hasattr(tree, "root_id")
    assert tree.root_id in tree.tree_node

    tree = PartitionTree(adj)
    tree.build_coding_tree(k=3, mode="v2")
    assert tree.root_id in tree.tree_node
    assert isinstance(tree.entropy(), float)
    assert isinstance(tree.entropy(tree.tree_node), float)

    # Unknown mode should preserve an already-built root.
    prev_root = tree.root_id
    tree.build_coding_tree(k=2, mode="unknown")
    assert tree.root_id == prev_root

    # Exercise helper methods that are not always reached by default trees.
    tree.build_sub_leaves(node_list=[0, 1], parent_vol=max(tree.VOL, 1.0))
    tree.build_root_down()
    assert isinstance(tree.root_down_delta(), tuple)

    filled = _make_select_output(
        assignment=torch.tensor([-1, 0, -1], dtype=torch.long),
        num_supernodes=1,
        s_inv_op="transpose",
    )
    assert filled.cluster_index.tolist() == [1, 0, 2]
    assert filled.num_supernodes == 3


def test_partition_tree_leaf_and_root_update_paths(monkeypatch):
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    tree = PartitionTree(adj)
    tree.build_coding_tree(k=2, mode="v1")

    state = {"leaf_calls": 0, "root_calls": 0}

    def fake_leaf_up(self):
        state["leaf_calls"] += 1
        if state["leaf_calls"] == 1:
            # Force else-branch first.
            return 0.5, {1: None}, {}
        return 0.0, {1: None}, {}

    def fake_root_down_delta(self):
        state["root_calls"] += 1
        root_down_dict = {
            99: PartitionTreeNode(
                ID=99,
                partition=[0],
                vol=1.0,
                g=1.0,
                children={0},
                parent=None,
                child_h=1,
            ),
            0: PartitionTreeNode(
                ID=0,
                partition=[0],
                vol=1.0,
                g=1.0,
                children=None,
                parent=99,
                child_h=0,
            ),
        }
        # First call larger than leaf_up; second call smaller.
        if state["root_calls"] == 1:
            return 0.2, 99, root_down_dict
        return 1.0, 99, root_down_dict

    def fake_leaf_up_update(self, _id_mapping, _leaf_up_dict):
        self.tree_node[self.root_id].child_h += 1

    def fake_root_down_update(self, _new_id, _root_down_dict):
        self.tree_node[self.root_id].child_h += 1

    monkeypatch.setattr(PartitionTree, "leaf_up", fake_leaf_up)
    monkeypatch.setattr(PartitionTree, "root_down_delta", fake_root_down_delta)
    monkeypatch.setattr(PartitionTree, "leaf_up_update", fake_leaf_up_update)
    monkeypatch.setattr(PartitionTree, "root_down_update", fake_root_down_update)
    monkeypatch.setattr(PartitionTree, "check_balance", lambda self, *_: None)
    monkeypatch.setattr(
        PartitionTree,
        "_build_k_tree",
        lambda self, *_args, **_kwargs: self.root_id
        if hasattr(self, "root_id")
        else max(self.tree_node),
    )

    # Ensure we enter the while-loop.
    tree.tree_node[tree.root_id].child_h = 2
    tree.build_coding_tree(k=5, mode="v2")
    assert tree.tree_node[tree.root_id].child_h >= 5


def test_leaf_up_update_and_root_down_update_branches():
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tree = PartitionTree(adj)
    tree.build_coding_tree(k=2, mode="v1")

    # None branch in leaf_up_update.
    root = tree.root_id
    parent_with_children = next(
        (
            nid
            for nid, node in tree.tree_node.items()
            if nid != root and node.children is not None
        ),
        None,
    )
    assert parent_with_children is not None
    tree.leaf_up_update({parent_with_children: None}, leaf_up_dict={})
    assert tree.tree_node[root].child_h >= 1

    # Non-none branch in leaf_up_update.
    host = parent_with_children
    new_leaf_id = 777
    h1_root = 700
    leaf_up_dict = {
        host: {
            h1_root: PartitionTreeNode(
                ID=h1_root,
                partition=[0],
                vol=1.0,
                g=1.0,
                children={new_leaf_id},
                parent=None,
                child_h=1,
            ),
            new_leaf_id: PartitionTreeNode(
                ID=new_leaf_id,
                partition=[0],
                vol=1.0,
                g=1.0,
                children=None,
                parent=h1_root,
                child_h=0,
            ),
        }
    }
    tree.leaf_up_update({host: h1_root}, leaf_up_dict=leaf_up_dict)
    assert new_leaf_id in tree.tree_node

    # root_down_update branch.
    root_down_new_id = 800
    injected_child = 801
    root_down_dict = {
        root_down_new_id: PartitionTreeNode(
            ID=root_down_new_id,
            partition=[0],
            vol=1.0,
            g=1.0,
            children={injected_child},
            parent=None,
            child_h=1,
        ),
        injected_child: PartitionTreeNode(
            ID=injected_child,
            partition=[0],
            vol=1.0,
            g=1.0,
            children=None,
            parent=root_down_new_id,
            child_h=0,
        ),
    }
    tree.root_down_update(root_down_new_id, root_down_dict)
    assert injected_child in tree.tree_node


def test_build_k_tree_branches():
    # No-edge case: triggers unmerged branch.
    pt = PartitionTree(np.zeros((2, 2)))
    nodes_dict = {
        0: PartitionTreeNode(ID=0, partition=[0], vol=1.0, g=1.0, child_h=1),
        1: PartitionTreeNode(ID=1, partition=[1], vol=1.0, g=1.0, child_h=1),
    }
    pt.adj_table = {0: set(), 1: set()}
    root = pt._build_k_tree(graph_vol=2.0, nodes_dict=nodes_dict, k=None)
    assert root in nodes_dict

    # No-edge case with child_h == 0: skip cmp_heap pushes in unmerged branch.
    pt = PartitionTree(np.zeros((2, 2)))
    nodes_dict = {
        0: PartitionTreeNode(ID=0, partition=[0], vol=1.0, g=1.0, child_h=0),
        1: PartitionTreeNode(ID=1, partition=[1], vol=1.0, g=1.0, child_h=0),
    }
    pt.adj_table = {0: set(), 1: set()}
    root = pt._build_k_tree(graph_vol=2.0, nodes_dict=nodes_dict, k=None)
    assert root in nodes_dict

    # Non-single partition branch (uses _cut_volume path).
    adj = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    pt = PartitionTree(adj)
    nodes_dict = {
        0: PartitionTreeNode(ID=0, partition=[0, 1], vol=2.0, g=1.0, child_h=1),
        1: PartitionTreeNode(ID=1, partition=[2], vol=1.0, g=1.0, child_h=1),
    }
    pt.adj_table = {0: {1}, 1: {0}}
    root = pt._build_k_tree(graph_vol=3.0, nodes_dict=nodes_dict, k=None)
    assert root in nodes_dict


def test_build_k_tree_k_loop_branches(monkeypatch):
    # Build a no-edge setup that enters the k-loop with cmp_heap populated.
    pt = PartitionTree(np.zeros((3, 3)))
    nodes_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={0},
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={1},
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[2],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={2},
        ),
    }
    pt.adj_table = {0: set(), 1: set(), 2: set()}

    # First run: force `continue` branch, then terminate.
    calls = {"depth": 0}
    import tgp.select.sep_select as ss

    original_child_depth = ss._child_tree_depth
    original_compress = ss._compress_node
    original_heapify = ss.heapq.heapify
    heapify_calls = {"count": 0}

    def fake_child_depth(node_dict_arg, _node_id):
        calls["depth"] += 1
        if calls["depth"] == 1:
            for node in node_dict_arg.values():
                node.child_h = 0
            return 0
        return 99

    def fake_compress(node_dict_arg, _node_id, _parent_id):
        for node in node_dict_arg.values():
            node.child_h = 0

    def wrapped_heapify(items):
        heapify_calls["count"] += 1
        return original_heapify(items)

    monkeypatch.setattr(ss, "_child_tree_depth", fake_child_depth)
    monkeypatch.setattr(ss, "_compress_node", fake_compress)
    monkeypatch.setattr(ss.heapq, "heapify", wrapped_heapify)
    root = pt._build_k_tree(graph_vol=3.0, nodes_dict=nodes_dict, k=1)
    assert root in nodes_dict

    # Second run: execute the block that updates cmp_heap entries + heapify.
    pt2 = PartitionTree(np.zeros((3, 3)))
    nodes_dict2 = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={0},
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={0, 1},
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[2],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={0, 2},
        ),
    }
    pt2.adj_table = {0: set(), 1: set(), 2: set()}

    monkeypatch.setattr(ss, "_child_tree_depth", lambda *_: 99)
    monkeypatch.setattr(ss, "_compress_node", fake_compress)
    heapify_calls["count"] = 0
    root = pt2._build_k_tree(graph_vol=3.0, nodes_dict=nodes_dict2, k=1)
    assert root in nodes_dict2
    assert heapify_calls["count"] > 0

    monkeypatch.setattr(ss, "_child_tree_depth", original_child_depth)
    monkeypatch.setattr(ss, "_compress_node", original_compress)
    monkeypatch.setattr(ss.heapq, "heapify", original_heapify)


def test_build_k_tree_priority_update_lines(monkeypatch):
    import tgp.select.sep_select as ss

    pt = PartitionTree(np.zeros((3, 3)))
    nodes_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={1, 2},
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={1},
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[2],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={2},
        ),
    }
    pt.adj_table = {0: set(), 1: set(), 2: set()}

    original_child_depth = ss._child_tree_depth
    original_compress = ss._compress_node
    original_heappush = ss.heapq.heappush

    def fake_child_depth(*_args, **_kwargs):
        return 99

    def fake_compress(node_dict_arg, _node_id, parent_id):
        if parent_id in node_dict_arg:
            node_dict_arg[parent_id].child_h = 0

    def wrapped_heappush(heap, item):
        original_heappush(heap, item)
        # cmp_heap stores [score, node_id, parent_id].
        if isinstance(item, list) and len(item) == 3:
            parent_id = item[2]
            original_heappush(heap, [item[0], parent_id, parent_id])

    monkeypatch.setattr(ss, "_child_tree_depth", fake_child_depth)
    monkeypatch.setattr(ss, "_compress_node", fake_compress)
    monkeypatch.setattr(ss.heapq, "heappush", wrapped_heappush)

    root = pt._build_k_tree(graph_vol=3.0, nodes_dict=nodes_dict, k=1)
    assert root in nodes_dict

    monkeypatch.setattr(ss, "_child_tree_depth", original_child_depth)
    monkeypatch.setattr(ss, "_compress_node", original_compress)
    monkeypatch.setattr(ss.heapq, "heappush", original_heappush)


def test_build_k_tree_cmp_heap_child_zero_continue(monkeypatch):
    import tgp.select.sep_select as ss

    pt = PartitionTree(np.zeros((3, 3)))
    nodes_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={0},
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={1},
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[2],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={2},
        ),
    }
    pt.adj_table = {0: set(), 1: set(), 2: set()}

    original_heappush = ss.heapq.heappush

    def fake_compress(node_dict_arg, _node_id, _parent_id):
        for node in node_dict_arg.values():
            node.child_h = 0

    def wrapped_heappush(heap, item):
        original_heappush(heap, item)
        # Keep a duplicate cmp entry for the same node_id so that after one pop,
        # an entry with entry[1] in `children` remains in the iteration loop.
        if isinstance(item, list) and len(item) == 3:
            original_heappush(heap, [item[0], item[1], item[2]])

    monkeypatch.setattr(ss, "_child_tree_depth", lambda *_: 99)
    monkeypatch.setattr(ss, "_compress_node", fake_compress)
    monkeypatch.setattr(ss.heapq, "heappush", wrapped_heappush)

    root = pt._build_k_tree(graph_vol=3.0, nodes_dict=nodes_dict, k=1)
    assert root in nodes_dict


def test_build_k_tree_cmp_heap_false_branches(monkeypatch):
    import tgp.select.sep_select as ss

    pt = PartitionTree(np.zeros((3, 3)))
    nodes_dict = {
        0: PartitionTreeNode(
            ID=0,
            partition=[0],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={1},
        ),
        1: PartitionTreeNode(
            ID=1,
            partition=[1],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={1},
        ),
        2: PartitionTreeNode(
            ID=2,
            partition=[2],
            vol=1.0,
            g=1.0,
            child_h=1,
            children={2},
        ),
    }
    pt.adj_table = {0: set(), 1: set(), 2: set()}

    original_heappush = ss.heapq.heappush

    def fake_child_depth(_node_dict_arg, node_id):
        # Trigger compression for the popped node, but force the two inner
        # conditions to be false in the cmp_heap update loop.
        if node_id == 0:
            return 99
        if node_id in {1, 3}:
            return 0
        return 99

    def fake_compress(node_dict_arg, _node_id, parent_id):
        # Drop root height below k so the loop exits after this iteration.
        if parent_id in node_dict_arg:
            node_dict_arg[parent_id].child_h = 0

    def wrapped_heappush(heap, item):
        original_heappush(heap, item)
        # Inject cmp entries where entry[1] == parent_id to hit line 809's
        # false branch.
        if isinstance(item, list) and len(item) == 3:
            parent_id = item[2]
            original_heappush(heap, [item[0], parent_id, parent_id])

    monkeypatch.setattr(ss, "_child_tree_depth", fake_child_depth)
    monkeypatch.setattr(ss, "_compress_node", fake_compress)
    monkeypatch.setattr(ss.heapq, "heappush", wrapped_heappush)

    root = pt._build_k_tree(graph_vol=3.0, nodes_dict=nodes_dict, k=1)
    assert root in nodes_dict


def test_build_coding_tree_defensive_flag_error(monkeypatch):
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tree = PartitionTree(adj)

    monkeypatch.setattr(
        PartitionTree, "_build_k_tree", lambda self, *_args, **_kwargs: 0
    )
    monkeypatch.setattr(PartitionTree, "check_balance", lambda self, *_: None)
    monkeypatch.setattr(PartitionTree, "leaf_up", lambda self: (1.0, {}, {}))
    monkeypatch.setattr(PartitionTree, "root_down_delta", lambda self: (0.0, None, {}))

    state = {"calls": 0}

    def force_invalid_flag(self, _id_mapping, _leaf_up_dict):
        state["calls"] += 1
        if state["calls"] > 3:
            raise RuntimeError("failed to inject invalid internal flag")
        frame = inspect.currentframe().f_back
        frame.f_locals["flag"] = 3
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))

    monkeypatch.setattr(PartitionTree, "leaf_up_update", force_invalid_flag)

    with pytest.raises(ValueError):
        tree.build_coding_tree(k=4, mode="v2")


def test_root_down_delta_and_leaf_up_entropy_paths(monkeypatch):
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    tree = PartitionTree(adj)
    tree.build_coding_tree(k=2, mode="v1")
    tree.tree_node[tree.root_id].children = {0, 1, 2}
    for nid in [0, 1, 2]:
        tree.tree_node[nid].parent = tree.root_id

    def fake_build_root_down(self):
        return (
            {
                0: PartitionTreeNode(
                    ID=0,
                    partition=[0],
                    vol=1.0,
                    g=1.0,
                    children=None,
                    parent=None,
                    child_h=0,
                ),
                1: PartitionTreeNode(
                    ID=1,
                    partition=[1],
                    vol=1.0,
                    g=1.0,
                    children=None,
                    parent=0,
                    child_h=0,
                ),
            },
            1.2,
        )

    monkeypatch.setattr(PartitionTree, "build_root_down", fake_build_root_down)
    monkeypatch.setattr(
        PartitionTree, "_build_k_tree", lambda self, *_args, **_kwargs: 0
    )
    monkeypatch.setattr(PartitionTree, "check_balance", lambda self, *_: None)
    monkeypatch.setattr(PartitionTree, "entropy", lambda self, *_: 0.2)

    delta, new_root, root_down_dict = tree.root_down_delta()
    assert new_root == 0
    assert isinstance(delta, float)
    assert isinstance(root_down_dict, dict)

    # leaf_up_entropy branch coverage (child_h == 1 and child_h != 1).
    tree.VOL = 10.0
    tree.tree_node[10] = PartitionTreeNode(ID=10, partition=[10], vol=5.0, g=2.0)
    tree.tree_node[11] = PartitionTreeNode(ID=11, partition=[11], vol=2.0, g=1.0)
    tree.tree_node[12] = PartitionTreeNode(ID=12, partition=[12], vol=3.0, g=1.5)
    sub_node_dict = {
        20: PartitionTreeNode(
            ID=20,
            partition=[11, 12],
            vol=1.0,
            g=1.0,
            children={11, 12},
            parent=None,
            child_h=2,
        ),
        11: PartitionTreeNode(
            ID=11,
            partition=[11],
            vol=1.0,
            g=0.2,
            children=None,
            parent=20,
            child_h=1,
        ),
        12: PartitionTreeNode(
            ID=12,
            partition=[12],
            vol=1.0,
            g=0.2,
            children=None,
            parent=20,
            child_h=2,
        ),
    }
    ent = tree.leaf_up_entropy(sub_node_dict, sub_root_id=20, node_id=10)
    assert np.isfinite(ent)


def test_leaf_up_large_partition_branch(monkeypatch):
    tree = PartitionTree(np.eye(3))
    tree.g_num_nodes = 3
    tree.leaves = [0, 1, 2]
    tree.tree_node[0] = PartitionTreeNode(ID=0, partition=[0], vol=1.0, g=1.0, parent=5)
    tree.tree_node[1] = PartitionTreeNode(ID=1, partition=[1], vol=1.0, g=1.0, parent=5)
    tree.tree_node[2] = PartitionTreeNode(ID=2, partition=[2], vol=1.0, g=1.0, parent=5)
    tree.tree_node[5] = PartitionTreeNode(
        ID=5,
        partition=[0, 1, 2],
        vol=3.0,
        g=1.0,
        children={0, 1, 2},
        parent=None,
        child_h=1,
    )

    monkeypatch.setattr(
        PartitionTree,
        "build_sub_leaves",
        lambda self, *_: (
            {
                55: PartitionTreeNode(
                    ID=55,
                    partition=[0, 1, 2],
                    vol=3.0,
                    g=1.0,
                    children=None,
                    parent=None,
                    child_h=1,
                )
            },
            1.0,
        ),
    )
    monkeypatch.setattr(
        PartitionTree, "_build_k_tree", lambda self, *_args, **_kwargs: 55
    )
    monkeypatch.setattr(PartitionTree, "check_balance", lambda self, *_: None)
    monkeypatch.setattr(PartitionTree, "leaf_up_entropy", lambda self, *_: 0.1)

    delta, id_mapping, h1_new_child_tree = tree.leaf_up()
    assert np.isfinite(delta)
    assert id_mapping[5] == 55
    assert 5 in h1_new_child_tree


if __name__ == "__main__":
    pytest.main([__file__])
