from .base_select import Select, SelectOutput
from .dp_select import DPSelect
from .edge_contraction_select import EdgeContractionSelect
from .eigenpool_select import EigenPoolSelect, eigenpool_select
from .graclus_select import GraclusSelect
from .identity_select import IdentitySelect
from .kmis_select import KMISSelect, degree_scorer
from .lapool_select import LaPoolSelect
from .maxcut_select import MaxCutScoreNet, MaxCutSelect
from .mlp_select import MLPSelect
from .ndp_select import NDPSelect
from .nmf_select import NMFSelect
from .topk_select import TopkSelect

select_functions = [
    "degree_scorer",
    "eigenpool_select",
]

select_classes = [
    "Select",
    "SelectOutput",
    "MLPSelect",
    "DPSelect",
    "EdgeContractionSelect",
    "EigenPoolSelect",
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
