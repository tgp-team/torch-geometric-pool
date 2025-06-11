import os

import torch
from torch_geometric.data import InMemoryDataset, download_url


class MultipartiteGraphDataset(InMemoryDataset):
    """The synthetic dataset for graph classification from the paper `"MaxCutPool:
    differentiable feature-aware Maxcut for pooling in graph neural networks"
    <https://arxiv.org/abs/2409.05100>`_ (Abate & Bianchi, ICLR 2025).
    """

    url = "https://zenodo.org/records/11617423/files/Multipartite.pkl?download=1"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        super(MultipartiteGraphDataset, self).__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return len(torch.unique(self.data.y))

    @property
    def raw_file_names(self):
        return ["Multipartite.pkl"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_list = torch.load(path)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
