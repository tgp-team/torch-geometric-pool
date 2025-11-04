from .csbm import CSBMDataset
from .expwl1 import EXPWL1Dataset
from .graph_classification_bench import GraphClassificationBench
from .gset import GsetDataset
from .multipartite_graph import MultipartiteGraphDataset
from .pygsp import PyGSPDataset

dataset_classes = [
    "EXPWL1Dataset",
    "GraphClassificationBench",
    "GsetDataset",
    "MultipartiteGraphDataset",
    "PyGSPDataset",
    "CSBMDataset",
]
