from .base_select import Select, SelectOutput
from .dense_select import DenseSelect
from .dp_select import DPSelect
from .edge_contraction_select import EdgeContractionSelect
from .graclus_select import GraclusSelect
from .identity_select import IdentitySelect
from .kmis_select import KMISSelect, degree_scorer
from .lapool_select import LaPoolSelect
from .maxcut_select import MaxCutScoreNet, MaxCutSelect
from .ndp_select import NDPSelect
from .nmf_select import NMFSelect
from .topk_select import TopkSelect

select_functions = [
    "degree_scorer",
]

select_classes = [
    "Select",
    "SelectOutput",
    "DenseSelect",
    "DPSelect",
    "EdgeContractionSelect",
    "GraclusSelect",
    "IdentitySelect",
    "LaPoolSelect",
    "KMISSelect",
    "MaxCutSelect",
    "NDPSelect",
    "NMFSelect",
    "TopkSelect",
]

__all__ = ["SelectOutput"] + select_classes + select_functions
