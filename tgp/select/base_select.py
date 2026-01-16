import copy
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from tgp.imports import SparseTensor
from tgp.utils.ops import connectivity_to_edge_index, get_assignments, pseudo_inverse


def cluster_to_s(
    cluster_index: Tensor,
    node_index: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    as_edge_index: bool = False,
    num_nodes: Optional[int] = None,
    num_supernodes: Optional[int] = None,
):
    r"""Converts a cluster assignment vector to a sparse assignment matrix.

    Args:
        cluster_index (~torch.Tensor):
            The cluster indices.
        node_index (~torch.Tensor, optional):
            The node indices. (default: :obj:`None`)
        weight (~torch.Tensor, optional):
            A weight vector, denoting the membership of a node to
            each cluster. (default: :obj:`None`)
        as_edge_index (bool, optional):
            If set to :obj:`True`, will return the edge indices of the assignment matrix.
            (default: :obj:`False`)
        num_nodes (int, optional):
            The number of nodes. (default: :obj:`None`)
        num_supernodes (int, optional):
            The number of clusters. (default: :obj:`None`)
    """
    if num_nodes is None:
        num_nodes = cluster_index.size(0)
    if node_index is None:
        node_index = torch.arange(
            num_nodes, dtype=torch.long, device=cluster_index.device
        )
    if as_edge_index:
        return torch.stack([node_index, cluster_index], dim=0), weight
    else:
        return SparseTensor(
            row=node_index,
            col=cluster_index,
            value=weight,
            sparse_sizes=(num_nodes, num_supernodes),
        )


# @torch.jit.script
@dataclass(init=False)
class SelectOutput:
    r"""The output of a :class:`~tgp.select.Select` method, which holds an assignment
    from selected nodes to their respective cluster(s).

    Args:
        node_index (~torch.Tensor):
            The indices of the selected nodes.
        num_nodes (int):
            The number of nodes.
        cluster_index (~torch.Tensor):
            The indices of the clusters each node in :obj:`node_index` is assigned to.
        num_supernodes (int):
            The number of clusters.
        weight (~torch.Tensor, optional):
            A weight vector, denoting the membership of a node to
            each cluster. (default: :obj:`None`)
    """

    s: Union[SparseTensor, Tensor]
    s_inv: Union[SparseTensor, Tensor] = None
    batch: Optional[Tensor] = None

    def __init__(
        self,
        s: Union[SparseTensor, Tensor] = None,
        s_inv: Union[SparseTensor, Tensor] = None,
        node_index: Tensor = None,
        num_nodes: int = None,
        cluster_index: Tensor = None,
        num_supernodes: int = None,
        weight: Optional[Tensor] = None,
        s_inv_op: Optional[str] = "transpose",
        batch: Optional[Tensor] = None,
        **extra_args,
    ):
        super().__init__()
        if isinstance(s, SparseTensor):  # Sparse assignment
            assert cluster_index is None, (
                "'cluster_index' cannot be set if 's' is not None"
            )
            assert node_index is None, "'node_index' cannot be set if 's' is not None"
            if weight is not None:
                s = s.set_value(weight)
            if num_nodes is not None or num_supernodes is not None:
                _N, _C = s.sparse_sizes()
                size = (num_nodes or _N, num_supernodes or _C)
                s = s.sparse_resize(size)
        elif isinstance(s, Tensor):  # Dense assignment
            assert cluster_index is None, (
                "'cluster_index' cannot be set if 's' is a dense Tensor"
            )
            assert node_index is None, (
                "'node_index' cannot be set if 's' is a dense Tensor"
            )
            assert num_nodes is None, (
                "'num_nodes' cannot be set if 's' is a dense Tensor"
            )
            assert num_supernodes is None, (
                "'num_supernodes' cannot be set if 's' is a dense Tensor"
            )
            assert weight is None, "'weight' cannot be set if 's' is a dense Tensor"
        elif s is None:  # Make sparse assignment from other data
            assert cluster_index is not None, (
                "'cluster_index' cannot be None if 's' is None"
            )

            s = cluster_to_s(
                cluster_index,
                node_index=node_index,
                num_supernodes=num_supernodes,
                num_nodes=num_nodes,
                weight=weight,
            )
        else:
            raise ValueError(
                "Either a sparse or dense assignment matrix is provided "
                "through 's' or a cluster assignment vector must be "
                "provided thorough 'cluster_index'."
            )

        self.s = s
        self.s_inv = s_inv
        if s_inv is None:
            self.set_s_inv(s_inv_op)

        self.batch = batch

        self._extra_args = set()
        for k, v in extra_args.items():
            setattr(self, k, v)
            self._extra_args.add(k)

    @property
    def is_expressive(self) -> bool:
        """Check if the assignment matrix is produced by an expressive pooling
        method. An assignment matrix is expressive if all rows sum to the same
        constant and that constant is non-zero.
        """
        row_sum = self.s.sum(dim=-1)

        if "mask" in self._extra_args:
            mask = getattr(self, "mask")
            row_sum = row_sum[mask]
        constant = row_sum[0]

        return torch.allclose(
            row_sum, constant.expand_as(row_sum)
        ) and not torch.allclose(
            constant, torch.tensor(0, dtype=constant.dtype, device=constant.device)
        )

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.s, SparseTensor)

    @property
    def is_dense(self) -> bool:
        return isinstance(self.s, Tensor)

    @property
    def is_dense_unbatched(self) -> bool:
        return self.is_dense and self.s.dim() == 2

    @property
    def is_dense_batched(self) -> bool:
        return isinstance(self.s, Tensor) and self.s.dim() == 3

    @property
    def num_nodes(self) -> int:
        return self.s.size(-2)

    @property
    def num_supernodes(self) -> int:
        return self.s.size(-1)

    @property
    def node_index(self) -> Optional[Tensor]:
        return self.s.storage.row() if self.is_sparse else None

    @property
    def cluster_index(self) -> Optional[Tensor]:
        return self.s.storage.col() if self.is_sparse else None

    @property
    def weight(self) -> Optional[Tensor]:
        return self.s.storage.value() if self.is_sparse else None

    def set_s_inv(self, method):
        if method == "transpose":
            if self.is_sparse:
                self.s_inv = self.s.t()
            else:
                self.s_inv = self.s.transpose(-1, -2)
        elif method == "inverse":
            self.s_inv = pseudo_inverse(self.s)
        else:
            raise ValueError()

    def __repr__(self):
        out = (
            f"{self.__class__.__name__}("
            f"num_nodes={self.num_nodes}, "
            f"num_supernodes={self.num_supernodes}"
        )
        if len(self._extra_args):
            out += f", extra={self._extra_args}"
        out += ")"
        return out

    def apply(self, func: Callable) -> "SelectOutput":
        r"""Applies the function :obj:`func` to both :obj:`s` and :obj:`s_inv`."""
        self.s = func(self.s)
        if self.s_inv is not None:
            self.s_inv = func(self.s_inv)
        return self

    def clone(self) -> "SelectOutput":
        r"""Performs a deep-copy of the object."""
        return copy.deepcopy(self)

    def to(self, device: Union[int, str], non_blocking: bool = False) -> "SelectOutput":
        r"""Performs tensor dtype and/or device conversion for both :obj:`s` and
        :obj:`s_inv`.
        """
        self.apply(lambda x: x.to(device=device, non_blocking=non_blocking))
        if self.batch is not None:
            self.batch = self.batch.to(device=device, non_blocking=non_blocking)
        return self

    def cpu(self) -> "SelectOutput":
        r"""Copies attributes to CPU memory for both :obj:`s` and :obj:`s_inv`."""
        self.apply(lambda x: x.cpu())
        if self.batch is not None:
            self.batch = self.batch.cpu()
        return self

    def cuda(
        self, device: Optional[Union[int, str]] = None, non_blocking: bool = False
    ) -> "SelectOutput":
        r"""Copies attributes to CUDA memory for both :obj:`s` and :obj:`s_inv`."""
        self.apply(lambda x: x.cuda(device, non_blocking=non_blocking))
        if self.batch is not None:
            self.batch = self.batch.cuda(device, non_blocking=non_blocking)
        return self

    def detach_(self) -> "SelectOutput":
        r"""Detaches attributes from the computation graph for both :obj:`s`
        and :obj:`s_inv`.
        """
        return self.apply(lambda x: x.detach_())

    def detach(self) -> "SelectOutput":
        r"""Detaches attributes from the computation graph by creating a new
        tensor for both :obj:`s` and :obj:`s_inv`.
        """
        return self.apply(lambda x: x.detach())

    def requires_grad_(self, requires_grad: bool = True) -> "SelectOutput":
        r"""Tracks gradient computation for both :obj:`s` and :obj:`s_inv`."""
        return self.apply(lambda x: x.requires_grad_(requires_grad=requires_grad))

    def assign_all_nodes(
        self,
        adj: Optional[Adj] = None,
        weight: Optional[Tensor] = None,
        max_iter: int = 5,
        batch: Optional[Tensor] = None,
        closest_node_assignment: bool = True,
    ) -> "SelectOutput":
        r"""Extends a sparse selection to assign ALL nodes to the selected supernodes.

        This method converts a sparse selection (where only a subset of nodes
        are initially selected, e.g. top-k selection) into a complete assignment where every node in the
        graph is assigned to one of the selected supernodes.

        Args:
            adj (~torch_geometric.typing.Adj, optional): Graph connectivity matrix.
                Can be an edge_index tensor of shape :math:`(2, E)` or SparseTensor.
                Required for :obj:`"closest_node"` strategy. (default: :obj:`None`)
            weight (~torch.Tensor, optional): Node-level weights for the assignment.
                Must have shape :math:`(N,)` where :math:`N` is the total number of nodes.
                Note: This is different from edge weights. (default: :obj:`None`)
            max_iter (int, optional): Maximum number of message passing iterations
                for the :obj:`"closest_node"` strategy. Higher values allow assignment
                of more distant nodes through graph connectivity. Must be :obj:`> 0`
                for :obj:`"closest_node"` strategy. (default: :obj:`5`)
            batch (~torch.Tensor, optional): Batch assignment vector of shape :math:`(N,)`
                indicating which graph each node belongs to. When provided, ensures
                nodes are only assigned to supernodes within the same graph.
                (default: :obj:`None`)
            closest_node_assignment (bool, optional): If True, assign unlabeled nodes to the
                closest supernode. If False, use random assignment to supernodes.
                (default: :obj:`True`)

        Returns:
            SelectOutput: A new SelectOutput with complete node-to-supernode assignments.
            The returned object has :obj:`num_nodes` assignments (one per node) and
            :obj:`num_supernodes` supernodes (same as the original selection).

        Raises:
            AssertionError: If :obj:`adj` is :obj:`None` for :obj:`"closest_node"` strategy.
            AssertionError: If :obj:`max_iter <= 0` for :obj:`"closest_node"` strategy.
            ValueError: If :obj:`weight` size doesn't match the number of nodes.
            ValueError: If :obj:`adj` has an invalid type.
            ValueError: If :obj:`strategy` is not recognized.

        Example:
            >>> # Convert sparse top-k selection to full assignment
            >>> # Assume we have a SelectOutput from top-k selection
            >>> sparse_output = topk_selector(x, edge_index)  # Only k nodes selected
            >>> print(sparse_output.node_index.size(0))  # k nodes
            >>> # Extend to assign all nodes using graph connectivity
            >>> full_output = sparse_output.assign_all_nodes(
            ...     adj=edge_index, closest_node_assignment=True, max_iter=5
            ... )
            >>> print(full_output.node_index.size(0))  # N nodes (all nodes)
            >>> print(full_output.num_supernodes)  # Still k supernodes
        """
        # Get the kept nodes indices from the original SelectOutput
        kept_nodes = self.node_index

        # If all nodes are already kept, no assignment is needed
        if len(kept_nodes) == self.num_nodes:
            return self

        if closest_node_assignment:
            assert adj is not None, "adj must be provided for closest_node_assignment"
            assert max_iter > 0, (
                "max_iter must be greater than 0 for closest_node_assignment"
            )

            # Convert adjacency to edge_index format if needed
            if isinstance(adj, SparseTensor):
                edge_index, _ = connectivity_to_edge_index(adj)
            elif isinstance(adj, Tensor):
                edge_index = adj
            else:
                raise ValueError(f"Invalid adjacency type: {type(adj)}")

            # Handle the weight parameter if provided
            if weight is not None:
                if weight.size(0) != self.num_nodes:
                    raise ValueError(
                        f"Weight tensor size ({weight.size(0)}) must match the number of nodes ({self.num_nodes})"
                    )

        # Use get_assignments with graph-aware assignment
        assignments = get_assignments(
            kept_nodes,
            edge_index=edge_index if closest_node_assignment else None,
            max_iter=max_iter if closest_node_assignment else 0,
            batch=batch,
        )

        # Create new SelectOutput with updated cluster assignments
        new_select_output = SelectOutput(
            cluster_index=assignments[1],
            s_inv_op=getattr(self, "s_inv_op", "transpose"),
            weight=weight,
        )

        # Copy any additional attributes from the original SelectOutput
        for attr_name in self._extra_args if hasattr(self, "_extra_args") else []:
            if hasattr(self, attr_name):
                setattr(new_select_output, attr_name, getattr(self, attr_name))

        return new_select_output


class Select(torch.nn.Module):
    r"""An abstract base class implementing a sparse :math:`\texttt{select}` operator
    that maps the nodes of an input graph to supernodes of the pooled one.

    It returns a :class:`~tgp.select.SelectOutput` containing the sparse
    supernode assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{N \times K}`.
    """

    is_dense_batched: bool = False

    def reset_parameters(self):
        pass

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
                The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
                (default: :obj:`None`)
            edge_index (~torch.Tensor, optional):
                The edge indices. Is a tensor of of shape  :math:`[2, E]`,
                where :math:`E` is the number of edges in the batch.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional):
                A vector of shape  :math:`[E]` or  :math:`[E, 1]` containing the weights of the edges.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.
                (default: :obj:`None`)
            num_nodes (int, optional):
                The total number of nodes of the graphs in the batch.
                (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
