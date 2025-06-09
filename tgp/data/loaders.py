from typing import Any, List, Optional, Sequence, Union

import torch.utils
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader.dataloader import Collater

from tgp.data.collate import collate, separate


class PooledBatch(Batch):
    r"""A custom :class:`~torch_geometric.data.Batch` class for handling graph data
    with pooled-graph data attributes in :tgp:`tgp`.

    This class extends :class:`~torch_geometric.data.Batch` to support a batch of graphs
    along with precomputed pooled representations. It stores additional information
    needed to reconstruct individual graph data objects and manage multiple levels of
    pooled data.
    """

    @classmethod
    def from_data_list(
        cls,
        data_list: List[BaseData],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> "PooledBatch":
        r"""Constructs a :class:`~tgp.data.PooledBatch` from a list of graph
        :class:`~torch_geometric.data.Data` objects.

        This method collates node and edge attributes, as well as any
        sub-:class:`~torch_geometric.data.Data` object storing pooled data, from each
        graph in :obj:`data_list` into a single :class:`~tgp.data.PooledBatch`. It
        handles attribute increments and batch assignments, and stores metadata
        required to separate individual graphs later.

        Args:
            data_list (List[~torch_geometric.data.data.BaseData]):
                A list of :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` objects to batch.
            follow_batch (Optional[List[str]]): Keys for which to create additional
                batch assignment vectors (e.g., node-level attributes to track).
            exclude_keys (Optional[List[str]]): Attributes to exclude from collation.

        Returns:
            :class:`PooledBatch`: A batched object containing all graphs from
                :obj:`data_list`, with :obj:`_slice_dict` and :obj:`_inc_dict`
                attributes set for reconstruction.
        """
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], PooledBatch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)  # type: ignore
        batch._slice_dict = slice_dict  # type: ignore
        batch._inc_dict = inc_dict  # type: ignore

        return batch

    def get_example(self, idx: int) -> BaseData:
        r"""Retrieves an individual graph data object from the batch.

        This method separates the batched data at the specified index :obj:`idx`, using
        the stored :obj:`_slice_dict` and :obj:`_inc_dict` to reconstruct the original
        :class:`~torch_geometric.data.data.BaseData` object.

        Args:
            idx (int): Index of the graph to extract from the batch.

        Returns:
            ~torch_geometric.data.data.BaseData: The reconstructed
                :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` object at position :obj:`idx`.

        Raises:
            RuntimeError: If the batch was not created via :func:`from_data_list`,
                making reconstruction impossible.
        """
        if not hasattr(self, "_slice_dict"):
            raise RuntimeError(
                "Cannot reconstruct 'Data' object from 'Batch' because "
                "'Batch' was not created via 'Batch.from_data_list()'"
            )

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=getattr(self, "_slice_dict"),
            inc_dict=getattr(self, "_inc_dict"),
            decrement=True,
        )

        return data


class PoolCollater(Collater):
    r"""A custom collate function for pooling dataloaders.

    This class extends :class:`~torch_geometric.loader.dataloader.Collater` to produce
    a :class:`~tgp.data.PooledBatch` when collating a list of
    :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` objects. It invokes
    :meth:`~tgp.data.PooledBatch.from_data_list` to perform the batching with optional
    :obj:`follow_batch` and :obj:`exclude_keys` arguments.
    """

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return PooledBatch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        return super().__call__(batch)


class PoolDataLoader(torch.utils.data.DataLoader):
    r"""A DataLoader for pooled graph datasets, returning :class:`PooledBatch` objects.

    This class extends :class:`~torch.utils.data.DataLoader` to integrate with
    :class:`PoolCollater`, automatically batching pooled graph data. It accepts
    :obj:`follow_batch` and :obj:`exclude_keys` parameters to propagate to the collate
    function.

    Args:
        dataset (Union[~torch.utils.data.Dataset, Sequence[~torch_geometric.data.BaseData], ~torch_geometric.data.datapipes.DatasetAdapter]):
            The source dataset from which to load graph data.
        batch_size (int): Number of graphs per batch. (default: ``1``)
        shuffle (bool): Whether to shuffle the data each epoch. (default: :obj:`False`)
        follow_batch (list, optional): Keys for which to create assignment vectors
            in the batch. (default: :obj:`None`)
        exclude_keys (list, optional): Attributes to exclude from collation.
            (default: :obj:`None`)
        **kwargs: Additional keyword arguments forwarded to
            `torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=PoolCollater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
