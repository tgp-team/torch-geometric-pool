from .asap import ASAPooling
from .asym_cheeger_cut import AsymCheegerCutPooling
from .bnpool import BNPool
from .bnpool_sparse import SparseBNPool
from .diffpool import DiffPool
from .dmon import DMoNPooling
from .edge_contraction import EdgeContractionPooling
from .graclus import GraclusPooling
from .hosc import HOSCPooling
from .just_balance import JustBalancePooling
from .kmis import KMISPooling
from .lapool import LaPooling
from .maxcut import MaxCutPooling
from .mincut import MinCutPooling
from .ndp import NDPPooling
from .nmf import NMFPooling
from .nopool import NoPool
from .pan import PANPooling
from .sag import SAGPooling
from .topk import TopkPooling

pooler_classes = [
    "ASAPooling",
    "AsymCheegerCutPooling",
    "BNPool",
    "DiffPool",
    "DMoNPooling",
    "EdgeContractionPooling",
    "GraclusPooling",
    "HOSCPooling",
    "LaPooling",
    "JustBalancePooling",
    "KMISPooling",
    "MaxCutPooling",
    "MinCutPooling",
    "NDPPooling",
    "NMFPooling",
    "NoPool",
    "PANPooling",
    "SAGPooling",
    "SparseBNPool",
    "TopkPooling",
]

pooler_map = {
    "asap": ASAPooling,
    "acc": AsymCheegerCutPooling,
    "bnpool": BNPool,
    "diff": DiffPool,
    "dmon": DMoNPooling,
    "ec": EdgeContractionPooling,
    "graclus": GraclusPooling,
    "hosc": HOSCPooling,
    "lap": LaPooling,
    "jb": JustBalancePooling,
    "kmis": KMISPooling,
    "maxcut": MaxCutPooling,
    "mincut": MinCutPooling,
    "ndp": NDPPooling,
    "nmf": NMFPooling,
    "nopool": NoPool,
    "pan": PANPooling,
    "sag": SAGPooling,
    "spbnpool": SparseBNPool,
    "topk": TopkPooling,
}


def get_pooler(pooler_name: str, **kwargs):
    """Return a pooling operator initialized with filtered **kwargs.

    Args:
        pooler_name (str): Name of the pooler.
        **kwargs: Additional keyword arguments to be passed to the
                  pooler constructor; irrelevant ones are discarded.

    Returns:
        A pooling layer instance corresponding to `pooler_name`.
    """
    pooler_name = pooler_name.lower()
    if pooler_name not in pooler_map:
        raise ValueError(
            f"Unknown pooler_name='{pooler_name}'. "
            f"Available poolers: {list(pooler_map.keys())}"
        )

    pooler_cls = pooler_map[pooler_name]
    signature = pooler_cls.get_signature()

    if signature.has_kwargs:
        return pooler_cls(**kwargs)

    # Filter out any kwargs that aren't in the signature:
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature.args}

    # Instantiate the pooler:
    return pooler_cls(**filtered_kwargs)
