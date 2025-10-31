import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    get_laplacian,
    sort_edge_index,
)

from tgp.src import SRCPooling


class NormalizeAdj(BaseTransform):
    r"""Transforms the adjacency matrix :math:`\mathbf{A}`
    by applying the following transformation:

    .. math::
        \mathbf{A} \to \mathbf{I} - \delta \mathbf{L}

    where :math:`\mathbf{L}` is the normalized Laplacian
    of the graph and :math:`\delta` is a scaling factor.

    Args:
        delta (int, optional):
            Scaling factor for the Laplacian.
            (default: :obj:`0.85`)
    """

    def __init__(self, delta: float = 0.85) -> None:
        self.delta = delta
        super().__init__()

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        r"""Applies the normalization transform to the graph while preserving edge attributes.

        The transform computes the normalized Laplacian and rescales it with :math:`-\delta`. It also handles self-loops and
        concatenates additional edge attributes if available. Duplicate entries are coalesced by summing their values.

        Args:
            data (~torch_geometric.data.Data):
                A Data object containing graph data.

        Returns:
            ~torch_geometric.data.Data: The transformed data object with updated edge_index, edge_weight, and (optionally) edge_attr.
        """
        assert data.edge_index is not None
        N = data.num_nodes

        edge_index, edge_weight = data.edge_index, data.edge_weight

        # Check how many edges have self loops
        self_loop_mask = edge_index[0] == edge_index[1]
        initial_self_loops = self_loop_mask.sum().item()

        # Get the symmetrically normalized Laplacian (I - D^-.5 A D^-.5) in sparse format
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization="sym", num_nodes=N
        )

        # Check if new self loops have been added
        new_self_loop_mask = edge_index[0] == edge_index[1]
        num_new_self_loops = new_self_loop_mask.sum().item() - initial_self_loops

        # Rescale the Laplacian weights by -delta
        edge_weight = -self.delta * edge_weight

        # Add self-loops representing the identity matrix
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=N
        )

        # Prepare edge attributes for coalescing
        if data.edge_attr is not None:
            if num_new_self_loops > 0:
                num_self_loops = (
                    2 * N
                )  # self loops from Laplacian and self loops from add_self_loops
            else:
                num_self_loops = N  # self loops only from add_self_loops

            # Create zero edge attributes for the self-loops
            attr_dim = data.edge_attr.size(1)
            self_loop_attr = torch.zeros(
                num_self_loops, attr_dim, device=data.edge_attr.device
            )

            # Concatenate original edge attributes and self-loop attributes
            edge_attr = torch.cat([data.edge_attr, self_loop_attr], dim=0)

        else:
            edge_attr = None

        # Prepare edge values for coalescing
        if edge_attr is not None:
            edge_weight = edge_weight.unsqueeze(
                1
            )  # Shape: [num_edges + num_self_loops, 1]
            edge_value = torch.cat(
                [edge_weight, edge_attr], dim=1
            )  # Shape: [num_edges + num_self_loops, 1 + attr_dim]
        else:
            edge_value = edge_weight  # Shape: [num_edges + num_self_loops]

        # Coalesce the sparse matrix to remove duplicate entries and sum their values
        edge_index, edge_value = coalesce(edge_index, edge_value, N)

        # Split edge_value back into edge_weight and edge_attr
        if edge_attr is not None:
            edge_weight = edge_value[:, 0]
            edge_attr = edge_value[:, 1:].to(data.edge_attr.dtype)
            data.edge_attr = edge_attr
        else:
            edge_weight = edge_value

        data.edge_index = edge_index
        data.edge_weight = edge_weight

        return data


class SortNodes(BaseTransform):
    """Sorts the nodes of a graph based on their labels."""

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        r"""Sorts the nodes of the graph according to their labels.

        The function sorts the node labels, reassigns node indices accordingly, and updates the associated attributes.
        If edge attributes exist, they are re-sorted based on the new node ordering.

        Args:
            data (~torch_geometric.data.Data):
                A Data object containing graph data with attributes
                :attr:`edge_index`, :attr:`y`, :attr:`x`, and optionally :attr:`edge_attr`.

        Returns:
            ~torch_geometric.data.Data: The data object with sorted nodes and updated attributes.
        """
        assert data.edge_index is not None
        assert data.y is not None
        y_sorted, sort_idx = torch.sort(data.y)
        edge_index_renamed = torch.empty_like(data.edge_index)
        for new_i in range(data.num_nodes):
            i = sort_idx[new_i]
            mask_i = data.edge_index == i
            edge_index_renamed[mask_i] = new_i

        data.x = data.x[sort_idx]
        data.y = y_sorted
        # sort edge_index_renamed in order to have edges ordered by source
        if data.edge_attr is not None:
            data.edge_index, (data.edge_weight, data.edge_attr) = sort_edge_index(
                edge_index_renamed, edge_attr=[data.edge_weight, data.edge_attr]
            )
        else:
            data.edge_index, data.edge_weight = sort_edge_index(
                edge_index_renamed, data.edge_weight
            )

        return data


class PreCoarsening(BaseTransform):
    r"""A PyG transform that precomputes a hierarchy of pooled (coarsened) graphs
    and attaches them to the input :class:`~torch_geometric.data.Data` object.

    Takes a pooling operator from :class:`~tgp.src.SRCPooling` to build
    a multi-level pooling hierarchy. The pooling operator should implement the
    function :meth:`~tgp.src.SRCPooling.precoarsening`, which computes the pooled graph
    from the original graph. The pooling operator must not be trainable,
    i.e., it should not have learnable parameters.
    The graph is recursively coarsened for :obj:`recursive_depth` levels.
    At each level, a coarsened adjacency matrix and, optionally, a pooled batch
    is computed. The result is stored as a list of intermediate pooled subgraphs
    in :class:`~torch_geometric.data.Data`, which downstream GNN models can consume.

    Args:
        pooler (~tgp.src.SRCPooling):
            A non-trainable pooling operator that implements the
            function :meth:`~tgp.src.SRCPooling.precoarsening`.
        input_key (str, optional):
            The key in the data object from which to read the graph data.
            If :obj:`None`, uses the default data object.
        output_key (str, optional):
            The key in the data object where the pooled graphs will be stored.
            Defaults to :obj:`"pooled_data"`.
        recursive_depth (int):
            The number of recursive coarsening levels. Must be greater than 0.
    """

    def __init__(
        self,
        pooler: SRCPooling,
        input_key: str = None,
        output_key: str = "pooled_data",
        recursive_depth: int = 1,
    ) -> None:
        super().__init__()
        assert isinstance(pooler, SRCPooling)
        assert not pooler.is_trainable, "The pooler must not be trainable."
        self.pooler = pooler
        self.input_key = input_key
        self.output_key = output_key
        assert recursive_depth > 0
        self.recursive_depth = recursive_depth

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        """Precomputes the select and connect operations for a graph."""
        data_obj = data if self.input_key is None else getattr(data, self.input_key)

        pooled_out = []
        for d in range(self.recursive_depth):
            data_pooled = self.pooler.precoarsening(
                edge_index=data_obj.edge_index,
                edge_weight=data_obj.edge_weight,
                batch=data_obj.batch,
                num_nodes=data_obj.num_nodes,
            )
            data_obj = data_pooled.as_data()
            pooled_out.append(data_obj)

        setattr(data, self.output_key, pooled_out)

        return data
