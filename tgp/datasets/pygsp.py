import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix

from tgp.imports import check_pygsp_available, pygsp


class PyGSPDataset(InMemoryDataset):
    """Torch dataset wrapper for the graphs in the `PyGSP library
    <https://pygsp.readthedocs.io/en/stable/>`_.

    .. admonition:: Optional dependency
       :class: warning

       This class requires `PyGSP <https://pygsp.readthedocs.io/en/stable/>`_ to be
       installed. You can install it using pip:

       .. code-block:: bash

          pip install pygsp
    """

    def __init__(
        self,
        root,
        name="Community",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        kwargs=None,
    ):
        self._GRAPHS = [
            "Graph",
            "Airfoil",
            "BarabasiAlbert",
            "Comet",
            "Community",
            "DavidSensorNet",
            "ErdosRenyi",
            "FullConnected",
            "Grid2d",
            "Logo",
            "LowStretchTree",
            "Minnesota",
            "Path",
            "RandomRegular",
            "RandomRing",
            "Ring",
            "Sensor",
            "StochasticBlockModel",
            "SwissRoll",
            "Torus",
        ]

        self._NNGRAPHS = [
            "NNGraph",
            "Bunny",
            "Cube",
            "ImgPatches",
            "Grid2dImgPatches",
            "Sphere",
            "TwoMoons",
        ]
        check_pygsp_available()
        # check if the graph is in the list of available graphs.
        if name not in self._GRAPHS and name not in self._NNGRAPHS:
            raise ValueError(
                f"Graph {name} not available in PyGSP. Available graphs are:\n{self._GRAPHS}\nand\n{self._NNGRAPHS}"
            )

        if name in self._GRAPHS:
            graph = getattr(pygsp.graphs, name)
        else:
            graph = getattr(pygsp.graphs.nngraphs, name)
        self.G = graph(**kwargs) if kwargs is not None else graph()

        if name in ["Community", "StochasticBlockModel"]:
            self.labels = torch.tensor(self.G.info["node_com"])
        else:
            self.labels = torch.zeros(self.G.N, dtype=torch.long)

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        if torch_geometric.__version__ > "2.4":
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        edge_index, edge_weights = from_scipy_sparse_matrix(self.G.W)

        # Set coords if the graph does not have them
        if not hasattr(self.G, "coords"):
            self.G.set_coordinates(kind="spring", seed=42)

        data_list = [
            Data(
                x=torch.tensor(self.G.coords.astype("float32")),
                edge_index=edge_index,
                edge_weight=edge_weights.float(),
                y=self.labels,
            )
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if torch_geometric.__version__ > "2.4":
            self.save(data_list, self.processed_paths[0])
        else:
            torch.save(self.collate(data_list), self.processed_paths[0])
