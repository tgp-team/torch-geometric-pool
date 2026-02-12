import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tgp.select import Select, SelectOutput
from tgp.utils import connectivity_to_edge_index
from tgp.utils.typing import SinvType


@dataclass
class _SubgraphData:
    """Container for a single graph extracted from a mini-batch."""

    node_ids: Tensor
    edge_index: Tensor
    edge_weight: Optional[Tensor]


class SEPSelect(Select):
    r"""Select operator for Structural Entropy Pooling (SEP).

    The selector builds a coding tree per graph and uses its depth-1 partitions
    as hard cluster assignments.

    Args:
        s_inv_op (~tgp.utils.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from
            the select matrix. (default: :obj:`"transpose"`)
    """

    def __init__(self, s_inv_op: SinvType = "transpose"):
        super().__init__()
        self.s_inv_op = s_inv_op

    def forward(
        self,
        x: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor, optional):
                Unused placeholder to keep interface compatibility.
            edge_index (~torch_geometric.typing.Adj):
                Graph connectivity.
            edge_weight (~torch.Tensor, optional):
                Edge weights of shape :math:`[E]` or :math:`[E, 1]`.
            batch (~torch.Tensor, optional):
                Batch vector assigning nodes to graphs.
            num_nodes (int, optional):
                Total number of nodes in the input batch.

        Returns:
            :class:`~tgp.select.SelectOutput`: Hard assignment from nodes to
            depth-1 SEP clusters.
        """
        return self.multi_level_select(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
            levels=1,
            **kwargs,
        )[0]  # Return only the leaves of the tree

    def _normalize_select_inputs(
        self,
        edge_index: Optional[Adj],
        edge_weight: Optional[Tensor],
        batch: Optional[Tensor],
        num_nodes: Optional[int],
    ) -> tuple[Tensor, Optional[Tensor], Tensor, int]:
        """Normalize connectivity and batch tensors for selection."""
        edge_index, edge_weight = connectivity_to_edge_index(edge_index, edge_weight)
        if num_nodes is None:
            num_nodes = (
                int(batch.numel()) if batch is not None else maybe_num_nodes(edge_index)
            )

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        elif batch.numel() != num_nodes:
            raise ValueError(
                f"Expected batch with {num_nodes} nodes, got {batch.numel()}."
            )

        return edge_index, edge_weight, batch, int(num_nodes)

    def _cluster_subgraph_hierarchy(
        self,
        subgraph: _SubgraphData,
        levels: int,
    ) -> tuple[list[Tensor], list[int]]:
        """Compute local SEP assignments for one graph across ``levels``."""
        if levels < 1:
            raise ValueError(f"'levels' must be >= 1, got {levels}.")

        num_nodes = int(subgraph.node_ids.numel())
        device = subgraph.node_ids.device
        if num_nodes == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return [empty for _ in range(levels)], [0 for _ in range(levels)]

        edge_index, edge_weight = remove_self_loops(
            subgraph.edge_index, subgraph.edge_weight
        )
        edge_index, edge_weight = to_undirected(
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=num_nodes,
        )

        if edge_index.numel() == 0:
            return _identity_hierarchy(
                num_nodes=num_nodes, levels=levels, device=device
            )

        adj = (
            to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        tree_nodes = _adj_mat_to_coding_tree(adj, tree_depth=levels + 1)

        if levels == 1:
            depth_one = _depth_one_assignment(
                tree_nodes=tree_nodes,
                num_nodes=num_nodes,
                device=device,
            )
            return [depth_one], [int(depth_one.max().item()) + 1]

        absolute_assignments = [
            _depth_assignment(
                tree_nodes=tree_nodes,
                num_nodes=num_nodes,
                depth=depth,
                device=device,
            )
            for depth in range(1, levels + 1)
        ]
        return _absolute_to_sequential_assignments(absolute_assignments)

    def _build_global_hierarchy(
        self,
        local_hierarchies: list[tuple[_SubgraphData, list[Tensor], list[int]]],
        num_nodes: int,
        levels: int,
        device: torch.device,
    ) -> list[SelectOutput]:
        """Merge per-graph local assignments into batched global assignments."""
        global_assignments: list[Tensor] = []
        global_num_supernodes: list[int] = []

        # Level 1: original nodes -> first-level supernodes.
        level_assignment = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        prev_offsets: list[int] = []
        cluster_offset = 0

        for subgraph, level_assignments, level_num_supernodes in local_hierarchies:
            local_assignment = level_assignments[0]
            level_assignment[subgraph.node_ids] = local_assignment + cluster_offset
            prev_offsets.append(cluster_offset)
            cluster_offset += level_num_supernodes[0]

        global_assignments.append(level_assignment)
        global_num_supernodes.append(cluster_offset)

        # Level d: supernodes(d-1) -> supernodes(d).
        for level_idx in range(1, levels):
            prev_num_nodes = global_num_supernodes[level_idx - 1]
            level_assignment = torch.full(
                (prev_num_nodes,),
                -1,
                dtype=torch.long,
                device=device,
            )

            next_offsets = []
            cluster_offset = 0
            for (
                _subgraph,
                level_assignments,
                level_num_supernodes,
            ), prev_offset in zip(local_hierarchies, prev_offsets):
                local_assignment = level_assignments[level_idx]
                local_size = local_assignment.numel()
                level_assignment[prev_offset : prev_offset + local_size] = (
                    local_assignment + cluster_offset
                )
                next_offsets.append(cluster_offset)
                cluster_offset += level_num_supernodes[level_idx]

            global_assignments.append(level_assignment)
            global_num_supernodes.append(cluster_offset)
            prev_offsets = next_offsets

        return [
            _make_select_output(
                assignment=assignment,
                num_supernodes=num_supernodes,
                s_inv_op=self.s_inv_op,
            )
            for assignment, num_supernodes in zip(
                global_assignments, global_num_supernodes
            )
        ]

    def multi_level_select(
        self,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        levels: int = 1,
        **kwargs,
    ) -> list[SelectOutput]:
        """Compute multiple sequential SEP selections from a single tree build."""
        if levels < 1:
            raise ValueError(f"'levels' must be >= 1, got {levels}.")

        edge_index, edge_weight, batch, num_nodes = self._normalize_select_inputs(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
        )

        if num_nodes == 0:
            return [
                _empty_select_output(edge_index.device, self.s_inv_op)
                for _ in range(levels)
            ]

        subgraphs = _split_subgraphs(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
        )
        local_hierarchies = [
            (
                subgraph,
                *self._cluster_subgraph_hierarchy(subgraph=subgraph, levels=levels),
            )
            for subgraph in subgraphs
        ]
        if len(local_hierarchies) == 0:
            raise RuntimeError("Could not split any non-empty subgraph.")

        outputs = self._build_global_hierarchy(
            local_hierarchies=local_hierarchies,
            num_nodes=num_nodes,
            levels=levels,
            device=edge_index.device,
        )
        if len(outputs) != levels:
            raise RuntimeError(
                f"Expected {levels} level outputs, found {len(outputs)}."
            )
        return outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _identity_hierarchy(
    num_nodes: int,
    levels: int,
    device: torch.device,
) -> tuple[list[Tensor], list[int]]:
    """Return deterministic identity mappings for edgeless hierarchies."""
    first_level = torch.arange(num_nodes, dtype=torch.long, device=device)
    assignments = [first_level]
    num_supernodes = [num_nodes]

    for _ in range(1, levels):
        prev_num_nodes = num_supernodes[-1]
        assignments.append(
            torch.arange(prev_num_nodes, dtype=torch.long, device=device)
        )
        num_supernodes.append(prev_num_nodes)

    return assignments, num_supernodes


def _make_select_output(
    assignment: Tensor,
    num_supernodes: int,
    s_inv_op: SinvType,
) -> SelectOutput:
    """Build a :class:`SelectOutput` and deterministically fill missing labels."""
    missing = torch.nonzero(assignment < 0, as_tuple=False).view(-1)
    if missing.numel() > 0:
        assignment[missing] = torch.arange(
            num_supernodes,
            num_supernodes + missing.numel(),
            device=assignment.device,
        )
        num_supernodes += int(missing.numel())

    return SelectOutput(
        node_index=torch.arange(assignment.numel(), device=assignment.device),
        num_nodes=int(assignment.numel()),
        cluster_index=assignment,
        num_supernodes=int(num_supernodes),
        s_inv_op=s_inv_op,
    )


def _empty_select_output(device: torch.device, s_inv_op: SinvType) -> SelectOutput:
    """Build an empty :class:`SelectOutput`."""
    empty = torch.empty(0, dtype=torch.long, device=device)
    return SelectOutput(
        node_index=empty,
        num_nodes=0,
        cluster_index=empty,
        num_supernodes=0,
        s_inv_op=s_inv_op,
    )


def _split_subgraphs(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    batch: Tensor,
    num_nodes: int,
) -> list[_SubgraphData]:
    """Split a batched graph into per-graph local COO representations.

    Unlike :func:`torch_geometric.utils.unbatch`, this function correctly
    filters edge-level tensors (e.g., :obj:`edge_weight`) with edge masks.
    """
    if batch.numel() == 0:
        return []

    out: list[_SubgraphData] = []
    batch_size = int(batch.max().item()) + 1

    for graph_id in range(batch_size):
        node_ids = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
        if node_ids.numel() == 0:
            continue

        if edge_index.numel() == 0:
            sub_edge_index = edge_index.new_empty((2, 0))
            sub_edge_weight = (
                edge_weight.new_empty((0,)) if edge_weight is not None else None
            )
        else:
            edge_mask = (batch[edge_index[0]] == graph_id) & (
                batch[edge_index[1]] == graph_id
            )
            sub_edge_index = edge_index[:, edge_mask]
            sub_edge_weight = (
                edge_weight[edge_mask] if edge_weight is not None else None
            )

            if sub_edge_index.numel() > 0:
                node_to_local = torch.full(
                    (num_nodes,),
                    -1,
                    dtype=torch.long,
                    device=edge_index.device,
                )
                node_to_local[node_ids] = torch.arange(
                    node_ids.numel(),
                    device=edge_index.device,
                )
                sub_edge_index = node_to_local[sub_edge_index]
            else:
                sub_edge_index = edge_index.new_empty((2, 0))

        out.append(
            _SubgraphData(
                node_ids=node_ids,
                edge_index=sub_edge_index,
                edge_weight=sub_edge_weight,
            )
        )

    return out


def _depth_one_assignment(
    tree_nodes: dict[int, dict],
    num_nodes: int,
    device: torch.device,
) -> Tensor:
    """Convert depth-1 tree nodes into a node-to-cluster assignment vector."""
    assignment = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    # Deterministic ordering: process depth-1 nodes by id.
    depth_one_nodes = [
        tree_nodes[node_id]
        for node_id in sorted(tree_nodes.keys())
        if tree_nodes[node_id]["depth"] == 1
    ]

    cluster_id = 0
    for node in depth_one_nodes:
        children = node.get("children") or []
        if not children:
            continue

        leaf_nodes = [tree_nodes[c].get("graphID", c) for c in children]
        assignment[leaf_nodes] = cluster_id
        cluster_id += 1

    # Isolated/uncovered nodes are assigned to singleton clusters.
    missing = torch.nonzero(assignment < 0, as_tuple=False).view(-1)
    if missing.numel() > 0:
        assignment[missing] = torch.arange(
            cluster_id,
            cluster_id + missing.numel(),
            device=device,
        )

    return assignment


def _depth_assignment(
    tree_nodes: dict[int, dict],
    num_nodes: int,
    depth: int,
    device: torch.device,
) -> Tensor:
    """Assign each original node to its ancestor cluster at a target depth."""
    assignment = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    for node_id in range(num_nodes):
        if node_id not in tree_nodes:
            raise RuntimeError(f"Leaf node {node_id} not found in coding tree.")

        node = tree_nodes[node_id]
        while node["depth"] < depth:
            parent_id = node["parent"]
            if parent_id is None:
                break
            node = tree_nodes[parent_id]

        if node["depth"] != depth:
            raise RuntimeError(
                f"Could not find ancestor at depth={depth} for node {node_id}."
            )
        assignment[node_id] = int(node["ID"])
    return assignment


def _relabel_contiguous(assignment: Tensor) -> tuple[Tensor, int]:
    """Relabel cluster ids to contiguous [0, K) in first-seen order."""
    relabeled = torch.empty_like(assignment)
    mapping = {}
    next_id = 0
    for idx, cluster_id in enumerate(assignment.tolist()):
        cluster_id = int(cluster_id)
        if cluster_id not in mapping:
            mapping[cluster_id] = next_id
            next_id += 1
        relabeled[idx] = mapping[cluster_id]
    return relabeled, next_id


def _absolute_to_sequential_assignments(
    absolute_assignments: list[Tensor],
) -> tuple[list[Tensor], list[int]]:
    """Convert original-node depth assignments into sequential level mappings."""
    if len(absolute_assignments) == 0:
        return [], []

    relabeled = []
    num_clusters = []
    for assignment in absolute_assignments:
        relabeled_assignment, k = _relabel_contiguous(assignment)
        relabeled.append(relabeled_assignment)
        num_clusters.append(k)

    sequential = [relabeled[0]]
    for depth_idx in range(1, len(relabeled)):
        prev_assignment = relabeled[depth_idx - 1]
        curr_assignment = relabeled[depth_idx]
        prev_k = num_clusters[depth_idx - 1]

        mapping = torch.full(
            (prev_k,), -1, dtype=torch.long, device=prev_assignment.device
        )
        for node_idx in range(prev_assignment.numel()):
            prev_cluster = int(prev_assignment[node_idx].item())
            curr_cluster = int(curr_assignment[node_idx].item())
            if mapping[prev_cluster] < 0:
                mapping[prev_cluster] = curr_cluster
            elif int(mapping[prev_cluster].item()) != curr_cluster:
                raise RuntimeError(
                    "Invalid hierarchy: a child cluster maps to multiple parents."
                )

        if torch.any(mapping < 0):
            raise RuntimeError("Invalid hierarchy: missing parent mapping.")
        sequential.append(mapping)

    return sequential, num_clusters


# -----------------------------------------------------------------------------
# Tree construction utilities
# -----------------------------------------------------------------------------


def _adj_mat_to_coding_tree(adj: np.ndarray, tree_depth: int) -> dict[int, dict]:
    """Build a coding tree from an adjacency matrix.

    Connected components are processed independently and then merged under a
    single synthetic root to keep a single tree representation per graph.
    """
    num_nodes = adj.shape[0]
    if num_nodes == 0:
        return {}

    n_components, labels = connected_components(
        csgraph=adj,
        directed=False,
        return_labels=True,
    )

    if n_components == 1:
        return _trans_to_tree(adj, tree_depth)

    trees = []
    for comp_id in range(n_components):
        sub_nodes = [int(u) for u in np.where(labels == comp_id)[0]]

        if len(sub_nodes) == 1:
            leaf = sub_nodes[0]
            nodes = [
                {
                    "ID": leaf,
                    "parent": f"{comp_id}_1_0",
                    "depth": 0,
                    "children": None,
                }
            ]
            for depth in range(1, tree_depth + 1):
                nodes.append(
                    {
                        "ID": f"{comp_id}_{depth}_0",
                        "parent": (
                            f"{comp_id}_{depth + 1}_0" if depth < tree_depth else None
                        ),
                        "depth": depth,
                        "children": [nodes[-1]["ID"]],
                    }
                )
            trees.append(nodes)
            continue

        sub_adj = adj[np.ix_(sub_nodes, sub_nodes)]
        tree = _trans_to_tree(sub_adj, tree_depth)
        nodes = [dict(node) for node in tree.values()]

        # Remap local ids back to component/global ids.
        remap = {i: sub_nodes[i] for i in range(len(sub_nodes))}
        for node in nodes:
            if node["depth"] > 0:
                remap[node["ID"]] = f"{comp_id}_{node['depth']}_{node['ID']}"

        for node in nodes:
            node["ID"] = remap[node["ID"]]
            if node["depth"] < tree_depth:
                node["parent"] = remap[node["parent"]]
            else:
                node["parent"] = None
            if node["children"] is not None:
                node["children"] = [remap[c] for c in node["children"]]

        trees.append(nodes)

    # Global id remapping by depth (leaves keep their original indices).
    id_map = {}
    for depth in range(tree_depth + 1):
        for nodes in trees:
            for node in nodes:
                if node["depth"] == depth:
                    id_map[node["ID"]] = len(id_map) if depth > 0 else node["ID"]

    tree = {}
    root_ids = []
    for nodes in trees:
        for node in nodes:
            new_node = dict(node)
            new_node["parent"] = (
                id_map[new_node["parent"]] if new_node["parent"] is not None else None
            )
            new_node["children"] = (
                [id_map[c] for c in new_node["children"]]
                if new_node["children"] is not None
                else None
            )
            new_node["ID"] = id_map[new_node["ID"]]
            tree[new_node["ID"]] = new_node

            if new_node["parent"] is None:
                root_ids.append(new_node["ID"])

    root_ids = sorted(set(root_ids))
    root_id = min(root_ids)
    root_children = list(
        itertools.chain.from_iterable(tree[rid]["children"] for rid in root_ids)
    )

    for rid in root_ids:
        tree.pop(rid)

    for child in root_children:
        tree[child]["parent"] = root_id

    tree[root_id] = {
        "ID": root_id,
        "parent": None,
        "children": root_children,
        "depth": tree_depth,
    }
    return tree


def _trans_to_tree(adj: np.ndarray, tree_depth: int) -> dict[int, dict]:
    tree = PartitionTree(adj_matrix=adj)
    tree.build_coding_tree(tree_depth)
    return _update_node(tree.tree_node)


def _update_depth(tree: dict[int, "PartitionTreeNode"]) -> None:
    """Populate :attr:`child_h` from leaves to root."""
    wait_update = [node_id for node_id, node in tree.items() if node.children is None]
    while wait_update:
        next_wait = set()
        for node_id in wait_update:
            node = tree[node_id]
            if node.children is None:
                node.child_h = 0
            else:
                first_child = next(iter(node.children))
                node.child_h = tree[first_child].child_h + 1

            if node.parent is not None:
                next_wait.add(node.parent)
        wait_update = list(next_wait)


def _update_node(tree: dict[int, "PartitionTreeNode"]) -> dict[int, dict]:
    """Reindex tree nodes by depth and id.

    This is equivalent to the legacy implementation but uses a precomputed
    dictionary instead of repeated :obj:`list.index` scans.
    """
    _update_depth(tree)

    depth_id_pairs = sorted((node.child_h, node.ID) for node in tree.values())
    pair_to_new_id = {pair: idx for idx, pair in enumerate(depth_id_pairs)}

    new_tree = {}
    for node in tree.values():
        child_h = node.child_h
        new_id = pair_to_new_id[(child_h, node.ID)]

        if node.parent is None:
            new_parent = None
        else:
            new_parent = pair_to_new_id[(child_h + 1, node.parent)]

        if node.children is None:
            new_children = None
        else:
            new_children = [
                pair_to_new_id[(child_h - 1, child_id)] for child_id in node.children
            ]

        new_tree[new_id] = {
            "ID": new_id,
            "partition": list(node.partition),
            "parent": new_parent,
            "children": new_children,
            "vol": node.vol,
            "g": node.g,
            "merged": node.merged,
            "child_h": child_h,
            "child_cut": node.child_cut,
            "depth": child_h,
        }

    return new_tree


# -----------------------------------------------------------------------------
# Partition tree core algorithm
# -----------------------------------------------------------------------------


def _id_generator():
    node_id = 0
    while True:
        yield node_id
        node_id += 1


def _graph_parse(adj_matrix: np.ndarray):
    """Parse dense adjacency into graph-level statistics.

    This vectorized version replaces the original O(N^2) Python loops.
    """
    g_num_nodes = int(adj_matrix.shape[0])

    node_vol = adj_matrix.sum(axis=1).tolist()
    vol = float(sum(node_vol))

    rows, cols = np.nonzero(adj_matrix)
    adj_table = {i: set() for i in range(g_num_nodes)}
    for row, col in zip(rows.tolist(), cols.tolist()):
        adj_table[row].add(col)

    return g_num_nodes, vol, node_vol, adj_table


def _cut_volume(adj_matrix: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute the total edge weight between two node partitions."""
    if p1.size == 0 or p2.size == 0:
        return 0.0
    return float(adj_matrix[np.ix_(p1, p2)].sum())


def _layer_first(node_dict: dict[int, "PartitionTreeNode"], start_id: int):
    """Breadth-first traversal over tree nodes."""
    queue = [start_id]
    while queue:
        node_id = queue.pop(0)
        yield node_id
        if node_dict[node_id].children:
            queue.extend(node_dict[node_id].children)


def _merge_nodes(
    new_id: int,
    id1: int,
    id2: int,
    cut_v: float,
    node_dict: dict[int, "PartitionTreeNode"],
) -> None:
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    vol = node_dict[id1].vol + node_dict[id2].vol
    g_val = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h, node_dict[id2].child_h) + 1

    node_dict[new_id] = PartitionTreeNode(
        ID=new_id,
        partition=new_partition,
        children={id1, id2},
        g=g_val,
        vol=vol,
        child_h=child_h,
        child_cut=cut_v,
    )
    node_dict[id1].parent = new_id
    node_dict[id2].parent = new_id


def _compress_node(
    node_dict: dict[int, "PartitionTreeNode"],
    node_id: int,
    parent_id: int,
) -> None:
    parent_child_h = node_dict[parent_id].child_h
    children = node_dict[node_id].children

    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(children)

    for child in children:
        node_dict[child].parent = parent_id

    compressed_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (parent_child_h - compressed_child_h) == 1:
        while True:
            max_child_h = max(
                node_dict[c].child_h for c in node_dict[parent_id].children
            )
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break


def _child_tree_depth(node_dict: dict[int, "PartitionTreeNode"], node_id: int) -> int:
    node = node_dict[node_id]
    depth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        depth += 1
    depth += node_dict[node_id].child_h
    return depth


def _compress_delta(node: "PartitionTreeNode", parent: "PartitionTreeNode") -> float:
    return node.child_cut * math.log(parent.vol / node.vol)


def _combine_delta(
    node1: "PartitionTreeNode",
    node2: "PartitionTreeNode",
    cut_v: float,
    graph_vol: float,
) -> float:
    v1, v2 = node1.vol, node2.vol
    g1, g2 = node1.g, node2.g
    v12 = v1 + v2
    return (
        (v1 - g1) * math.log(v12 / v1, 2)
        + (v2 - g2) * math.log(v12 / v2, 2)
        - 2 * cut_v * math.log(graph_vol / v12, 2)
    ) / graph_vol


@dataclass
class PartitionTreeNode:
    """Node used by the SEP coding tree optimizer."""

    ID: int
    partition: list[int]
    vol: float
    g: float
    children: Optional[set[int]] = None
    parent: Optional[int] = None
    child_h: int = 0
    child_cut: float = 0.0
    merged: bool = False


class PartitionTree:
    """Internal tree optimizer used by SEPSelect."""

    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = _graph_parse(
            adj_matrix
        )
        self.id_gen = _id_generator()
        self.leaves = []
        self.build_leaves()

    def build_leaves(self) -> None:
        for vertex in range(self.g_num_nodes):
            node_id = next(self.id_gen)
            vol = self.node_vol[vertex]
            self.tree_node[node_id] = PartitionTreeNode(
                ID=node_id,
                partition=[vertex],
                g=vol,
                vol=vol,
            )
            self.leaves.append(node_id)

    def build_sub_leaves(self, node_list, parent_vol):
        subgraph_node_dict = {}
        ori_entropy = 0

        for vertex in node_list:
            ori_entropy += -(self.tree_node[vertex].g / self.VOL) * math.log2(
                self.tree_node[vertex].vol / parent_vol
            )

            sub_neighbors = set()
            vol = 0
            for vertex_n in node_list:
                cut_val = self.adj_matrix[vertex, vertex_n]
                if cut_val != 0:
                    vol += cut_val
                    sub_neighbors.add(vertex_n)

            subgraph_node_dict[vertex] = PartitionTreeNode(
                ID=vertex,
                partition=[vertex],
                g=vol,
                vol=vol,
            )
            self.adj_table[vertex] = sub_neighbors

        return subgraph_node_dict, ori_entropy

    def build_root_down(self):
        root_children = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_entropy = 0
        graph_vol = self.tree_node[self.root_id].vol

        for node_id in root_children:
            node = self.tree_node[node_id]
            ori_entropy += -(node.g / graph_vol) * math.log2(node.vol / graph_vol)

            new_neighbors = set()
            for neigh in self.adj_table[node_id]:
                if neigh in root_children:
                    new_neighbors.add(neigh)
            self.adj_table[node_id] = new_neighbors

            subgraph_node_dict[node_id] = PartitionTreeNode(
                ID=node_id,
                partition=node.partition,
                vol=node.vol,
                g=node.g,
                children=node.children,
            )

        return subgraph_node_dict, ori_entropy

    def entropy(self, node_dict=None):
        if node_dict is None:
            node_dict = self.tree_node

        ent = 0
        for node_id, node in node_dict.items():
            if node.parent is None:
                continue
            parent = node_dict[node.parent]
            ent += -(node.g / self.VOL) * math.log2(node.vol / parent.vol)
        return ent

    def _build_k_tree(self, graph_vol, nodes_dict, k=None):
        min_heap = []
        cmp_heap = []
        node_ids = nodes_dict.keys()
        new_id = None

        for i in node_ids:
            for j in self.adj_table[i]:
                if j <= i:
                    continue

                n1 = nodes_dict[i]
                n2 = nodes_dict[j]
                if len(n1.partition) == 1 and len(n2.partition) == 1:
                    cut_v = self.adj_matrix[n1.partition[0], n2.partition[0]]
                else:
                    cut_v = _cut_volume(
                        self.adj_matrix,
                        p1=np.array(n1.partition),
                        p2=np.array(n2.partition),
                    )

                diff = _combine_delta(nodes_dict[i], nodes_dict[j], cut_v, graph_vol)
                heapq.heappush(min_heap, (diff, i, j, cut_v))

        unmerged_count = len(node_ids)
        while unmerged_count > 1:
            if len(min_heap) == 0:
                break

            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue

            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True

            new_id = next(self.id_gen)
            _merge_nodes(new_id, id1, id2, cut_v, nodes_dict)

            self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            for neigh in self.adj_table[new_id]:
                self.adj_table[neigh].add(new_id)

            if nodes_dict[id1].child_h > 0:
                heapq.heappush(
                    cmp_heap,
                    [_compress_delta(nodes_dict[id1], nodes_dict[new_id]), id1, new_id],
                )
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(
                    cmp_heap,
                    [_compress_delta(nodes_dict[id2], nodes_dict[new_id]), id2, new_id],
                )

            unmerged_count -= 1

            for neigh in self.adj_table[new_id]:
                if nodes_dict[neigh].merged:
                    continue

                n1 = nodes_dict[neigh]
                n2 = nodes_dict[new_id]
                cut_v = _cut_volume(
                    self.adj_matrix,
                    np.array(n1.partition),
                    np.array(n2.partition),
                )
                new_diff = _combine_delta(nodes_dict[neigh], n2, cut_v, graph_vol)
                heapq.heappush(min_heap, (new_diff, neigh, new_id, cut_v))

        root = new_id

        if unmerged_count > 1:
            unmerged_nodes = {i for i, node in nodes_dict.items() if not node.merged}
            new_child_h = max(nodes_dict[i].child_h for i in unmerged_nodes) + 1

            new_id = next(self.id_gen)
            nodes_dict[new_id] = PartitionTreeNode(
                ID=new_id,
                partition=list(node_ids),
                children=unmerged_nodes,
                vol=graph_vol,
                g=0,
                child_h=new_child_h,
            )

            for node_id in unmerged_nodes:
                nodes_dict[node_id].merged = True
                nodes_dict[node_id].parent = new_id
                if nodes_dict[node_id].child_h > 0:
                    heapq.heappush(
                        cmp_heap,
                        [
                            _compress_delta(nodes_dict[node_id], nodes_dict[new_id]),
                            node_id,
                            new_id,
                        ],
                    )
            root = new_id

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, parent_id = heapq.heappop(cmp_heap)
                if _child_tree_depth(nodes_dict, node_id) <= k:
                    continue

                children = nodes_dict[node_id].children
                _compress_node(nodes_dict, node_id, parent_id)
                if nodes_dict[root].child_h == k:
                    break

                for entry in cmp_heap:
                    if entry[1] == parent_id:
                        if _child_tree_depth(nodes_dict, parent_id) > k:
                            entry[0] = _compress_delta(
                                nodes_dict[entry[1]], nodes_dict[entry[2]]
                            )

                    if entry[1] in children:
                        if nodes_dict[entry[1]].child_h == 0:
                            continue
                        if _child_tree_depth(nodes_dict, entry[1]) > k:
                            entry[2] = parent_id
                            entry[0] = _compress_delta(
                                nodes_dict[entry[1]],
                                nodes_dict[parent_id],
                            )

                heapq.heapify(cmp_heap)

        return root

    def check_balance(self, node_dict, root_id):
        root_children = set(node_dict[root_id].children)
        for child in root_children:
            if node_dict[child].child_h == 0:
                self.single_up(node_dict, child)

    def single_up(self, node_dict, node_id):
        new_id = next(self.id_gen)
        parent_id = node_dict[node_id].parent

        node_dict[new_id] = PartitionTreeNode(
            ID=new_id,
            partition=node_dict[node_id].partition,
            parent=parent_id,
            children={node_id},
            vol=node_dict[node_id].vol,
            g=node_dict[node_id].g,
        )

        node_dict[node_id].parent = new_id
        node_dict[parent_id].children.remove(node_id)
        node_dict[parent_id].children.add(new_id)

        node_dict[new_id].child_h = node_dict[node_id].child_h + 1

        self.adj_table[new_id] = self.adj_table[node_id]
        for neigh in self.adj_table[node_id]:
            self.adj_table[neigh].add(new_id)

    def root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0, None, None

        subgraph_node_dict, ori_entropy = self.build_root_down()
        graph_vol = self.tree_node[self.root_id].vol
        new_root = self._build_k_tree(
            graph_vol=graph_vol, nodes_dict=subgraph_node_dict, k=2
        )
        self.check_balance(subgraph_node_dict, new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self, sub_node_dict, sub_root_id, node_id):
        ent = 0
        for sub_node_id in _layer_first(sub_node_dict, sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g
                continue

            if sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                parent = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / parent.vol)
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                parent = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / parent.vol)

        return ent

    def leaf_up(self):
        h1_ids = {self.tree_node[leaf].parent for leaf in self.leaves}
        h1_new_child_tree = {}
        id_mapping = {}
        delta = 0

        for node_id in h1_ids:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition

            if len(sub_nodes) <= 2:
                id_mapping[node_id] = None
                continue

            sub_graph_vol = candidate_node.vol - candidate_node.g
            subgraph_node_dict, ori_entropy = self.build_sub_leaves(
                sub_nodes,
                candidate_node.vol,
            )

            sub_root = self._build_k_tree(
                graph_vol=sub_graph_vol,
                nodes_dict=subgraph_node_dict,
                k=2,
            )
            self.check_balance(subgraph_node_dict, sub_root)

            new_entropy = self.leaf_up_entropy(subgraph_node_dict, sub_root, node_id)
            delta += ori_entropy - new_entropy

            h1_new_child_tree[node_id] = subgraph_node_dict
            id_mapping[node_id] = sub_root

        delta = delta / self.g_num_nodes
        return delta, id_mapping, h1_new_child_tree

    def leaf_up_update(self, id_mapping, leaf_up_dict):
        for node_id, h1_root in id_mapping.items():
            if h1_root is None:
                children = set(self.tree_node[node_id].children)
                for child in children:
                    self.single_up(self.tree_node, child)
                continue

            h1_dict = leaf_up_dict[node_id]
            self.tree_node[node_id].children = h1_dict[h1_root].children
            for h1_child in h1_dict[h1_root].children:
                assert h1_child not in self.tree_node
                h1_dict[h1_child].parent = node_id
            h1_dict.pop(h1_root)
            self.tree_node.update(h1_dict)

        self.tree_node[self.root_id].child_h += 1

    def root_down_update(self, new_id, root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id

        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode="v2"):
        if k == 1:
            return

        if mode == "v1" or k is None:
            self.root_id = self._build_k_tree(self.VOL, self.tree_node, k=k)

        elif mode == "v2":
            self.root_id = self._build_k_tree(self.VOL, self.tree_node, k=2)
            self.check_balance(self.tree_node, self.root_id)

            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2

            flag = 0
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                    root_down_delta, new_id, root_down_dict = self.root_down_delta()
                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                elif flag == 2:
                    root_down_delta, new_id, root_down_dict = self.root_down_delta()
                else:
                    raise ValueError

                if leaf_up_delta < root_down_delta:
                    flag = 2
                    self.root_down_update(new_id, root_down_dict)
                else:
                    flag = 1
                    self.leaf_up_update(id_mapping, leaf_up_dict)

                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items():
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[
                                    root_down_id
                                ].children

        count = 0
        for _ in _layer_first(self.tree_node, self.root_id):
            count += 1
        assert len(self.tree_node) == count
