from io import StringIO
from os import path

import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx


class GsetDataset(InMemoryDataset):
    """Torch dataset wrapper for the `Gset dataset
    <http://web.stanford.edu/~yyye/yyye/Gset>`_.

    The dataset is mainly used to evaluate the performance of MAXCUT algorithms.
    """

    base_url = "http://web.stanford.edu/~yyye/yyye/Gset/"

    def parse_graph(self, data):
        """Parse the graph data and create a NetworkX graph."""
        # Create a buffer from the data string
        buf = StringIO(data)

        # Read the first line separately which contains the number of nodes and edges
        num_nodes, num_edges = map(int, buf.readline().strip().split())

        # Create a graph with the given number of nodes
        G = nx.DiGraph() if self._DIRECTED else nx.Graph()
        G.add_nodes_from(range(1, num_nodes + 1))

        # Read the rest of the lines, which contain the edges
        for line in buf:
            parts = line.strip().split()
            if len(parts) == 3:
                u, v, weight = map(int, parts)
                # Add an edge to the graph
                G.add_edge(u, v, weight=weight)

        return G

    def __init__(
        self,
        root,
        name="G1",
        directed=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        self._DIRECTED = directed

        self._GRAPHS = [
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
            "G6",
            "G7",
            "G8",
            "G9",
            "G10",
            "G11",
            "G12",
            "G13",
            "G14",
            "G15",
            "G16",
            "G17",
            "G18",
            "G19",
            "G20",
            "G21",
            "G22",
            "G23",
            "G24",
            "G25",
            "G26",
            "G27",
            "G28",
            "G29",
            "G30",
            "G31",
            "G32",
            "G33",
            "G34",
            "G35",
            "G36",
            "G37",
            "G38",
            "G39",
            "G40",
            "G41",
            "G42",
            "G43",
            "G44",
            "G45",
            "G46",
            "G47",
            "G48",
            "G49",
            "G50",
            "G51",
            "G52",
            "G53",
            "G54",
            "G55",
            "G56",
            "G57",
            "G58",
            "G59",
            "G60",
            "G61",
            "G62",
            "G63",
            "G64",
            "G65",
            "G66",
            "G67",
            "G70",
            "G72",
            "G77",
            "G81",
        ]

        if name not in self._GRAPHS:
            raise ValueError(f"Invalid graph name: {name}")
        self.file_name = name

        super(GsetDataset, self).__init__(
            root, transform, pre_transform, pre_filter, **kwargs
        )

        if torch_geometric.__version__ > "2.4":
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.file_name

    @property
    def processed_file_names(self):
        return "{}.pt".format(self.file_name)

    def download(self):
        download_url("{}{}".format(self.base_url, self.file_name), self.raw_dir)

    def process(self):
        with open(path.join(self.raw_dir, self.raw_file_names), "rb") as f:
            data = f.read().decode("utf-8")
            graph = self.parse_graph(data)
            pyg_graph = from_networkx(graph)

            rnd_feats = torch.randn(pyg_graph.num_nodes, 32)

            data_list = [
                Data(
                    # x=pyg_graph.x,
                    x=rnd_feats,
                    edge_index=pyg_graph.edge_index,
                    edge_attr=pyg_graph.weight,
                    num_nodes=pyg_graph.num_nodes,
                )
            ]

            if self.pre_filter is not None:
                data = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data = [self.pre_transform(data) for data in data_list]

            if torch_geometric.__version__ > "2.4":
                self.save(data_list, self.processed_paths[0])
            else:
                torch.save(self.collate(data_list), self.processed_paths[0])
