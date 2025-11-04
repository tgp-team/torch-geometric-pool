import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected


@dataclass(frozen=True)
class CSBMParameters:
    num_graphs: int
    num_nodes_per_class: int
    dbar: float
    lam: float
    gamma: float
    mu: Optional[float]
    only_sbm: bool
    structured_ratio: float
    seed: int


def _sbm_edge_probabilities(
    num_nodes_per_class: int,
    dbar: float,
    lam: float,
) -> Tuple[float, float]:
    r"""Return in/out edge probabilities for the SBM.

    :obj:`"dbar"` and :obj:`"lam"` must be chosen such that
    :math:`\bar{d} \geq \lambda \sqrt{\bar{d}}`.

    Args:
        num_nodes_per_class (int): Number of nodes in each block.
        dbar (float): Average degree.
        lam (float): signal strength for the SBM structure

    Returns:
        Tuple[float, float]: (pin, pout) edge probabilities within and between blocks.
    """
    n = 2 * num_nodes_per_class
    sqrt_dbar = math.sqrt(dbar)
    pin = (dbar + lam * sqrt_dbar) / n
    pout = (dbar - lam * sqrt_dbar) / n
    for name, value in {"pin": pin, "pout": pout}.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name}={value:.4f} is outside [0, 1]; "
                "please adjust dbar and lam to produce valid probabilities."
            )
    return pin, pout


def _generate_sbm_graph(
    num_nodes_per_class: int,
    pin: float,
    pout: float,
    rng: np.random.Generator,
) -> Tuple[nx.Graph, np.ndarray]:
    """Sample a (possibly disconnected) SBM graph and its block assignments.

    Args:
        num_nodes_per_class (int): Number of nodes in each block.
        pin (float): Edge probability within blocks.
        pout (float): Edge probability between blocks.
        rng (np.random.Generator): Random number generator.

    Returns:
        Tuple[nx.Graph, np.ndarray]: The generated SBM graph and block assignments.
    """
    n = 2 * num_nodes_per_class
    blocks = np.concatenate(
        [
            np.zeros(num_nodes_per_class, dtype=np.int64),
            np.ones(num_nodes_per_class, dtype=np.int64),
        ]
    )
    rng.shuffle(blocks)

    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        same_block = blocks[i] == blocks[i + 1 :]
        probs = np.where(same_block, pin, pout)
        mask = rng.random(n - i - 1) < probs
        adjacency[i, i + 1 :][mask] = True
    adjacency |= adjacency.T  # Make symmetric

    graph = nx.from_numpy_array(adjacency.astype(int))
    return graph, blocks


def _generate_connected_sbm(
    num_nodes_per_class: int,
    pin: float,
    pout: float,
    rng: np.random.Generator,
    max_attempts: int = 128,
) -> Tuple[nx.Graph, np.ndarray]:
    """Sample a connected SBM graph, retrying when necessary (ie, rejection sampling).

    Args:
        num_nodes_per_class (int): Number of nodes in each block.
        pin (float): Edge probability within blocks.
        pout (float): Edge probability between blocks.
        rng (np.random.Generator): Random number generator.
        max_attempts (int): Maximum number of sampling attempts.

    Returns:
        Tuple[nx.Graph, np.ndarray]: A connected SBM graph and its block assignments.
    """
    for _ in range(1, max_attempts + 1):
        graph, blocks = _generate_sbm_graph(num_nodes_per_class, pin, pout, rng)
        if nx.is_connected(graph):
            return graph, blocks
    raise RuntimeError(
        "Failed to sample a connected SBM graph after "
        f"{max_attempts} attempts. Please verify the parameters."
    )


def _generate_random_graph_with_degree_sequence(
    degrees: Sequence[int],
    rng: np.random.Generator,
    max_attempts: int = 128,
) -> nx.Graph:
    """Generate a connected random graph matching a target degree sequence.

    Args:
        degrees (Sequence[int]): Target degree sequence.
        rng (np.random.Generator): Generator for the random seed.
        max_attempts (int): Maximum number of attempts to sample a connected graph.

    Returns:
        nx.Graph: A connected random graph with the specified degree sequence.
    """
    for _ in range(max_attempts):
        seed = int(rng.integers(0, 2**32 - 1))
        candidate = nx.random_degree_sequence_graph(degrees, tries=100, seed=seed)
        candidate = nx.Graph(candidate)  # enforce simple graph
        candidate.remove_edges_from(nx.selfloop_edges(candidate))
        if nx.is_connected(candidate):
            return candidate
    raise RuntimeError(
        "Unable to sample a connected random graph with the provided degree sequence."
    )


def _degree_vector(graph: nx.Graph) -> np.ndarray:
    """Return the node degrees ordered by node index."""
    return np.array([graph.degree(i) for i in range(graph.number_of_nodes())])


def _gaussian_features(
    n: int,
    feature_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Noise features with unit variance per dimension."""
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    return rng.normal(size=(n, feature_dim)) / math.sqrt(feature_dim)


def _gmm_features(
    blocks: np.ndarray,
    mu: float,
    feature_dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian mixture features aligned with the community structure.

    Args:
        blocks (np.ndarray): Block assignments for each node.
        mu (float): Signal strength.
        feature_dim (int): Dimensionality of the features.
        rng (np.random.Generator): Random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Node features and the cluster centroid.
    """
    n = blocks.size
    centroid = rng.normal(size=feature_dim) / math.sqrt(feature_dim)
    noise = _gaussian_features(n, feature_dim, rng)
    signal_scale = math.sqrt(mu / n)
    signed_blocks = (blocks * 2 - 1).reshape(-1, 1)  # map {0,1} -> {-1,1}
    features = signal_scale * signed_blocks * centroid + noise
    return features, signal_scale * centroid


def _build_data_object(
    graph: nx.Graph,
    features: np.ndarray,
    label: int,
    node_gt: Optional[np.ndarray] = None,
    centroid: Optional[np.ndarray] = None,
) -> Data:
    """Create a PyG Data object from a NetworkX graph and feature matrix.

    Args:
        graph (nx.Graph): Input graph.
        features (np.ndarray): Node feature matrix.
        label (int): Graph label.
        node_gt (Optional[np.ndarray]): Optional ground truth node labels.
        centroid (Optional[np.ndarray]): Optional centroid of the feature clusters.

    Returns:
        ~~torch_geometric.data.Data: The constructed PyG Data object.
    """
    edges = list(graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=graph.number_of_nodes())

    x = torch.from_numpy(features.astype(np.float32))
    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    if node_gt is not None:
        data.node_gt = torch.from_numpy(node_gt.astype(np.int64))
    if centroid is not None:
        data.centroid = torch.from_numpy(centroid.astype(np.float32))

    return data


class CSBMDataset(InMemoryDataset):
    """Community SBM dataset for graph classification tasks in PyG format.

    Args:
        root (str): Root directory for the dataset.
        num_graphs (int): Number of graphs to generate.
        num_nodes_per_class (int): Number of nodes in each community.
        dbar (float): Average degree of the graphs.
        lam (float): Signal strength for the SBM structure.
        gamma (float): Ratio of number of nodes to feature dimension.
        mu (Optional[float]): Signal strength for node features (ignored if only_sbm is :obj:`True`).
        only_sbm (bool): If True, generate only SBM structure without node features.
        structured_ratio (float): Proportion of graphs with community structure.
        seed (int): Random seed for reproducibility.
        transform (callable, optional): A function/transform that takes in an
            :obj:`~torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`~torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to disk.
        pre_filter (callable, optional): A function that takes in an
            :obj:`~torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset.
    """

    def __init__(
        self,
        root: str,
        num_graphs: int = 1024,
        num_nodes_per_class: int = 50,
        dbar: float = 10.0,
        lam: float = 1.0,
        gamma: float = 4.0,
        mu: Optional[float] = 4.0,
        only_sbm: bool = False,
        structured_ratio: float = 0.5,
        seed: int = 80,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ):
        self.params = CSBMParameters(
            num_graphs=num_graphs,
            num_nodes_per_class=num_nodes_per_class,
            dbar=dbar,
            lam=lam,
            gamma=gamma,
            mu=mu,
            only_sbm=only_sbm,
            structured_ratio=structured_ratio,
            seed=seed,
        )
        if not 0.0 <= structured_ratio <= 1.0:
            raise ValueError("structured_ratio must be in the interval [0, 1].")
        if not only_sbm and mu is None:
            raise ValueError("mu must be provided when only_sbm is False.")

        if force_reload:
            self._purge_processed_files(root)

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Sequence[str]:
        return []

    @property
    def processed_file_names(self) -> Sequence[str]:
        p = self.params
        suffix = (
            f"graphs{p.num_graphs}_ncls{p.num_nodes_per_class}_d{p.dbar:.2f}"
            f"_lam{p.lam:.3f}_gamma{p.gamma:.3f}_mu{p.mu if p.mu is not None else 'None'}"
            f"_onlysbm{int(p.only_sbm)}_ratio{p.structured_ratio:.2f}_seed{p.seed}"
        )
        safe_suffix = suffix.replace(".", "p")
        return [f"csbm_{safe_suffix}.pt"]

    def _purge_processed_files(self, root: str) -> None:
        from pathlib import Path

        processed_dir = Path(root) / "processed"
        for filename in self.processed_file_names:
            path = processed_dir / filename
            if path.exists():
                path.unlink()

    def download(self):
        # Dataset is generated locally, no download required.
        return

    def process(self):
        p = self.params
        rng = np.random.default_rng(p.seed)
        pin, pout = _sbm_edge_probabilities(p.num_nodes_per_class, p.dbar, p.lam)

        total_nodes = 2 * p.num_nodes_per_class
        feature_dim = 1 if p.only_sbm else max(1, int(round(total_nodes / p.gamma)))

        num_structured = int(round(p.num_graphs * p.structured_ratio))
        num_structured = min(p.num_graphs, max(0, num_structured))
        num_unstructured = p.num_graphs - num_structured

        data_list = []

        # Unstructured graphs: same degree profile but without block signal.
        for _ in range(num_unstructured):
            sbm_graph, _ = _generate_connected_sbm(
                p.num_nodes_per_class, pin, pout, rng
            )  # to work exactly with a degree distribution drawn from an SBM
            degrees = _degree_vector(sbm_graph).tolist()
            random_graph = _generate_random_graph_with_degree_sequence(degrees, rng)

            if p.only_sbm:
                features = np.asarray(_degree_vector(random_graph), dtype=np.float32)
                features = features.reshape(-1, 1)
            else:
                features = _gaussian_features(total_nodes, feature_dim, rng)

            data = _build_data_object(random_graph, features, label=0)
            data_list.append(data)

        # Structured graphs: SBM structure with informative node features.
        for _ in range(num_structured):
            graph, blocks = _generate_connected_sbm(
                p.num_nodes_per_class, pin, pout, rng
            )
            if p.only_sbm:
                features = _degree_vector(graph).astype(np.float32).reshape(-1, 1)
                centroid = None
            else:
                features, centroid = _gmm_features(
                    blocks, float(p.mu), feature_dim, rng
                )

            data = _build_data_object(
                graph,
                features,
                label=1,
                node_gt=blocks,
                centroid=centroid,
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
