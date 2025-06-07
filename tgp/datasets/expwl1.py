import os
import pickle

import torch
from torch_geometric.data import InMemoryDataset, download_url


class EXPWL1Dataset(InMemoryDataset):
    """The synthetic dataset for graph classification from the paper
    `"The expressive power of pooling in graph neural networks"
    <https://arxiv.org/abs/2304.01575>`_ (Bianchi & Lachi, NeurIPS 2023).
    """

    url = (
        "https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs/"
        "raw/main/data/EXPWL1/raw/EXPWL1.pkl"
    )

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        super(EXPWL1Dataset, self).__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["EXPWL1.pkl"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        with open(os.path.join(self.root, "raw/EXPWL1.pkl"), "rb") as f:
            data_list = pickle.load(f)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
