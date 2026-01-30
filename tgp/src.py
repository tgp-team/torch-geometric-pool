from collections.abc import Iterator
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_sparse import SparseTensor

from tgp.connect import Connect
from tgp.lift import Lift
from tgp.reduce import Reduce, dense_global_reduce, global_reduce
from tgp.select import Select, SelectOutput
from tgp.utils import Signature, check_and_filter_edge_weights, foo_signature
from tgp.utils.typing import ReduceType


@dataclass
class PoolingOutput:
    r"""The pooling output of a model of class :class:`~tgp.src.SRCPooling`.

    Args:
        x (~torch.Tensor): The pooled node features.
        edge_index (~torch.Tensor): The edge indices of the pooled graph.
        edge_weight (~torch.Tensor, optional): The edge features of the coarsened
            graph. (default: :obj:`None`)
        batch (~torch.Tensor, optional): The batch vector of the pooled nodes.
        so (:class:`~tgp.select.SelectOutput`): The selection output. (default: :obj:`None`)
        loss (Optional[Dict], optional): The loss dictionary. (default: :obj:`None`)
    """

    x: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None
    batch: Optional[Tensor] = None
    so: Optional[SelectOutput] = None
    loss: Optional[Dict] = None

    def __repr__(self) -> str:
        return (
            f"PoolingOutput(so={[self.so.num_nodes, self.so.num_supernodes] if self.so is not None else None}, "
            f"x={[*self.x.shape] if self.x is not None else None}, "
            f"edge_index={[*self.edge_index.shape] if self.edge_index is not None else None}, "
            f"edge_weight={[*self.edge_weight.shape] if self.edge_weight is not None else None}, "
            f"batch={[*self.batch.shape] if self.batch is not None else None}, "
            f"loss={list(self.loss.keys()) if self.loss is not None else None})"
        )

    def __iter__(self) -> Iterator:
        return iter(self.__dict__.values())

    @property
    def has_loss(self):
        r"""Returns :obj:`True` if the pooling output has a loss."""
        return bool(isinstance(self.loss, dict) and len(self.loss) > 0)

    def get_loss_value(self, name: str = None) -> Union[float, List[float]]:
        r"""Returns the value of the loss with name :obj:`name` or all losses.
        If the pooling output does not have a loss, it returns :obj:`0`.

        Args:
            name (str, optional): The name of the loss to return. If :obj:`None`, returns all losses.
                (default: :obj:`None`)

        Returns:
            Union[float, List[float]]: The value of the loss :obj:`name` or all losses.
        """
        if not self.has_loss:
            return 0
        if name is None:
            return [v for v in self.loss.values()]
        return self.loss[name]

    def as_data(self):
        r"""Converts the pooling output to a :class:`~torch_geometric.data.Data` object.

        Returns:
            ~torch_geometric.data.Data: The pooling output as a Data object.
        """
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            batch=self.batch,
            so=self.so,
            num_nodes=self.so.num_supernodes if self.so is not None else None,
        )


class SRCPooling(torch.nn.Module):
    r"""A base class for pooling layers based on the paper
    `"Understanding Pooling in Graph Neural Networks" <https://arxiv.org/abs/1905.05178>`_
    (Grattarola et al., TNNLS 2022). Each pooler should inherit from this class.

    :class:`~tgp.src.SRCPooling` decomposes a pooling layer into three components:

    + :class:`~tgp.select.Select` defines how input nodes map to supernodes.
    + :class:`~tgp.reduce.Reduce` defines how input node features are aggregated.
    + :class:`~tgp.lift.Lift` defines how pooled node features are un-pooled.
    + :class:`~tgp.connect.Connect` decides how the supernodes are connected to each other.

    This class should return an object of type :obj:`~tgp.src.PoolingOutput`.

    Args:
        selector (:class:`~tgp.select.Select`): The node selection operator.
        reducer (:class:`~tgp.reduce.Reduce`): The node feature aggregation operator.
        lifter (:class:`~tgp.lift.Lift`): The node feature un-pooling operator.
        connector (:class:`~tgp.connect.Connect`): The edge connection operator.
        cached (bool, optional): If set to :obj:`True`, will cache the
            :class:`~tgp.select.Select` output and the :class:`~tgp.connect.Connect`
            output. (default: :obj:`False`)
        node_dim (int, optional): The dimension of the node features.
            (default: :obj:`-2`)
    """

    def __init__(
        self,
        selector: Select = None,
        reducer: Reduce = None,
        lifter: Lift = None,
        connector: Connect = None,
        cached: bool = False,
        node_dim: int = -2,
    ):
        super().__init__()
        self.selector = selector
        self.reducer = reducer
        self.lifter = lifter
        self.connector = connector
        self.node_dim = node_dim
        self.cached = cached
        self._so_cached = None
        self._pooled_edge_index = None
        self._pooled_edge_weight = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.selector.reset_parameters()
        self.reducer.reset_parameters()
        self.lifter.reset_parameters()
        self.connector.reset_parameters()

    def select(
        self,
        **kwargs,
    ) -> SelectOutput:
        r"""Calls the :class:`~tgp.select.Select` operator.

        Returns:
            An object of type :class:`~tgp.select.SelectOutput` containing the
            mapping from nodes to supernodes :math:`\mathbf{S} \in \mathbb{R}^{N \times K}`.
        """
        if self.selector is not None:
            if self._so_cached is not None:
                return self._so_cached
            so = self.selector(
                **kwargs,
            )
            if self.cached:
                self._so_cached = so
            return so
        raise NotImplementedError

    def reduce(
        self,
        **kwargs,
    ) -> Tensor:
        r"""Calls the :class:`~tgp.reduce.Reduce` operator.

        Returns:
            The pooled supernode features :math:`\mathbf{X}_{\text{pool}}`.
        """
        if self.reducer is not None:
            return self.reducer(**kwargs)
        raise NotImplementedError

    def lift(self, **kwargs) -> Adj:
        r"""Calls the :class:`~tgp.lift.Lift` operator.

        Returns:
            The un-pooled node features :math:`\mathbf{X}_{\text{lift}} \approx \mathbf{X}`.
        """
        if self.lifter is not None:
            return self.lifter(**kwargs)
        raise NotImplementedError

    def connect(
        self,
        **kwargs,
    ) -> Tuple[Adj, Optional[Tensor]]:
        r"""Calls the :class:`~tgp.connect.Connect` operator.

        Returns:
            The adjacency matrix of the coarse graph :math:`\mathbf{A}_{\text{pool}}`.
        """
        if self.connector is not None:
            if self._pooled_edge_index is not None:
                return self._pooled_edge_index, self._pooled_edge_weight
            pooled_edge_index, pooled_edge_weight = self.connector(**kwargs)
            if self.cached:
                self._pooled_edge_index = pooled_edge_index
                self._pooled_edge_weight = pooled_edge_weight
            return pooled_edge_index, pooled_edge_weight
        raise NotImplementedError

    def preprocessing(
        self, x: Tensor, edge_index: Tensor, **kwargs
    ) -> Tuple[Adj, Tensor, Optional[Tensor]]:
        """Preprocess inputs, if needed."""
        return x, edge_index, None

    def global_pool(
        self,
        x: Tensor,
        reduce_op: ReduceType = "sum",
        batch: Optional[Tensor] = None,
        size: Optional[int] = None,
    ) -> Tensor:
        r"""Global pooling operation.

        It is just a wrapper for :func:`~tgp.reduce.global_reduce`.
        """
        return global_reduce(x, reduce_op, batch, size, self.node_dim)

    @property
    def is_dense_batched(self) -> bool:
        """Returns :obj:`True` if the pooler is a dense pooling method."""
        if self.selector is not None:
            return self.selector.is_dense_batched
        raise NotImplementedError

    @property
    def has_loss(self) -> bool:
        r"""Returns :obj:`True` if the pooler has implemented the
        :meth:`~tgp.SRCPooling.compute_loss` method.
        """
        return self.compute_loss.__qualname__.split(".")[0] != "SRCPooling"

    @property
    def is_trainable(self) -> bool:
        r"""Returns :obj:`True` if any parameter belonging to the pooler or any
        of its registered sub-modules is trainable.
        """
        return any(p.requires_grad for p in self.parameters())

    def compute_loss(self, *args, **kwargs) -> Optional[dict]:
        """Compute loss function."""
        return None

    def clear_cache(self):
        r"""Clear the caching done by :math:`\texttt{select}` and :math:`\texttt{connect}`."""
        self._so_cached = None
        self._pooled_edge_index = None
        self._pooled_edge_weight = None

    @property
    def is_precoarsenable(self) -> bool:
        r"""Returns :obj:`True` if the pooler is precoarsenable."""
        if isinstance(self, Precoarsenable):
            return not self.is_trainable
        else:
            return False

    @classmethod
    def get_signature(cls) -> Signature:
        """Get signature of the pooler's :obj:`__init__` function."""
        return foo_signature(cls)

    @classmethod
    def get_forward_signature(cls) -> Signature:
        """Get signature of the pooler's :obj:`forward` function."""
        return foo_signature(cls.forward)

    @staticmethod
    def data_transforms():
        """Transforms to apply to the dataset before passing it to the model."""
        return None

    def __repr__(self) -> str:
        out = [f"{self.__class__.__name__}("]
        out.append(f"\tselect={self.selector}")
        out.append(f"\treduce={self.reducer}")
        out.append(f"\tlift={self.lifter}")
        out.append(f"\tconnect={self.connector}")
        for k, v in self.extra_repr_args().items():
            out.append(f"\t{k}={v}")
        out.append(")")
        return "\n".join(out)

    def extra_repr_args(self) -> dict:
        """Add extra arguments to :meth:`~tgp.SRCPooling.__repr__`."""
        return {}


class DenseSRCPooling(SRCPooling):
    r"""A base class for *dense* pooling layers that extends :class:`~tgp.src.SRCPooling`.

    It provides a preprocessing function that transform a batch of graphs in
    sparse representation into a batch of dense graphs.
    It also specifies how to perform global pooling through the
    :func:`~tgp.reduce.dense_global_reduce` function.

    Args:
        selector (:class:`~tgp.select.Select`): The *dense* :math:`\texttt{select}` operator.
        reducer (:class:`~tgp.reduce.Reduce`): The *dense* :math:`\texttt{reduce}` operator.
        lifter (:class:`~tgp.lift.Lift`): The *dense* :math:`\texttt{lift}` operator.
        connector (:class:`~tgp.connect.Connect`): The *dense* :math:`\texttt{connect}` operator.
        cached (bool, optional): If set to :obj:`True`, will cache the
            :class:`~tgp.select.Select` output and the :class:`~tgp.connect.Connect`
            output. (default: :obj:`None`)
        node_dim (int, optional): The dimension of the node features.
            (default: :obj:`-2`)
        adj_transpose (bool, optional):
            If :obj:`True`, the preprocessing step and
            the :class:`~tgp.connect.DenseConnect` operation returns transposed
            adjacency matrices, so that they could be passed "as is" to the dense
            message-passing layers.
            (default: :obj:`True`)
    """

    def __init__(
        self,
        selector: Select = None,
        reducer: Reduce = None,
        lifter: Lift = None,
        connector: Connect = None,
        cached: bool = False,
        node_dim: int = -2,
        adj_transpose: bool = False,
    ):
        super().__init__(
            selector=selector,
            reducer=reducer,
            lifter=lifter,
            connector=connector,
            cached=cached,
            node_dim=node_dim,
        )
        self.adj_transpose = adj_transpose
        self.preprocessing_cache = None

    def preprocessing(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        max_num_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Adj, Tensor, Tensor]:
        r"""Preprocess inputs for dense pooling methods.

        Transform a batch of graphs in sparse representation into a batch of graphs
        with dense representation.

        Args:
            x (~torch.Tensor):
                The node features.
                A tensor of shape :math:`[N, F]`, where :math:`N` is the total number
                of nodes in the batch and :math:`F` is the number of node features.
            edge_index (~torch.Tensor):
                The edge indices.
                A tensor of of shape  :math:`[2, E]`,
                where :math:`E` is the number of edges in the batch.
            edge_weight (~torch.Tensor, optional):
                A vector of shape  :math:`[E]` or :math:`[E, 1]` containing the weights
                of the edges.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.
                (default: :obj:`None`)
            max_num_nodes (int, optional):
                The maximum number of nodes of a graph in the batch.
                (default: :obj:`None`)
            batch_size (int, optional):
                The number of graphs in the batch.
                (default: :obj:`None`)
            use_cache (bool, optional):
                If :obj:`True`, it stores the preprocessed edge_index.
                (default: :obj:`False`)

        Returns:
            (x, adj, mask) tuple[~torch.Tensor, ~torch.Tensor, ~torch.Tensor]:
            :obj:`x` is the batched node features with
            shape :math:`[B, N_\text{max}, F]`, where :math:`N_\text{max}` is the size
            of the largest graph in the batch.
            :obj:`adj` is the tensor of shape :math:`[B, N_\text{max}, N_\text{max}]`
            containing the batched and dense adjacency matrices.
            :obj:`mask` is the tensor of shape :math:`[B, N_\text{max}, F]`
            indicating which nodes in the batch are valid or padded.
        """
        if use_cache and self.preprocessing_cache is not None:
            adj = self.preprocessing_cache

        else:
            if isinstance(edge_index, SparseTensor):
                row, col, edge_weight = edge_index.coo()
                edge_index = torch.stack([row, col])
            else:
                edge_weight = check_and_filter_edge_weights(edge_weight)

            adj = to_dense_adj(
                edge_index=edge_index,
                edge_attr=edge_weight,
                max_num_nodes=max_num_nodes,
                batch=batch,
                batch_size=batch_size,
            )

            if self.adj_transpose:
                adj = adj.transpose(-1, -2)

            if use_cache:
                self.preprocessing_cache = adj

        x, mask = to_dense_batch(
            x=x, batch=batch, max_num_nodes=max_num_nodes, batch_size=batch_size
        )

        return x, adj, mask

    def global_pool(
        self,
        x: Tensor,
        reduce_op: ReduceType = "sum",
        batch: Optional[Tensor] = None,
        size: Optional[int] = None,
    ) -> Tensor:
        r"""Global pooling operation for dense pooling methods.

        It is just a wrapper for :func:`~tgp.reduce.dense_global_reduce`.
        """
        return dense_global_reduce(x, reduce_op, self.node_dim)


class Precoarsenable:
    def precoarsening(
        self,
        **kwargs,
    ) -> PoolingOutput:
        """Precompute a coarsened graph from the original graph.
        Must be implemented by the poolers that support precoarsening.
        """
        raise NotImplementedError("Precoarsening is not supported by this pooler.")


class BasePrecoarseningMixin(Precoarsenable):
    r"""A mixin class for pooling layers that implements the
    pre-coarsening strategy.
    """

    def precoarsening(
        self,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> PoolingOutput:
        so = self.select(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
            **kwargs,
        )
        batch_pooled = self.reducer.reduce_batch(select_output=so, batch=batch)
        edge_index_pooled, edge_weight_pooled = self.connector(
            so=so, edge_index=edge_index, edge_weight=edge_weight
        )
        return PoolingOutput(
            edge_index=edge_index_pooled,
            edge_weight=edge_weight_pooled,
            batch=batch_pooled,
            so=so,
        )
