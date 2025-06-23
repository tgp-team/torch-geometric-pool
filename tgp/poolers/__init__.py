from .asap import ASAPooling
from .asym_cheeger_cut import AsymCheegerCutPooling
from .diffpool import DiffPool
from .dmon import DMoNPooling
from .edge_contraction import EdgeContractionPooling
from .graclus import GraclusPooling
from .hosc import HOSCPooling
from .just_balance import JustBalancePooling
from .kmis import KMISPooling
from .lapool import LaPooling
from .mincut import MinCutPooling
from .ndp import NDPPooling
from .nmf import NMFPooling
from .pan import PANPooling
from .sag import SAGPooling
from .topk import TopkPooling

pooler_classes = [
    "ASAPooling",
    "AsymCheegerCutPooling",
    "DiffPool",
    "DMoNPooling",
    "EdgeContractionPooling",
    "GraclusPooling",
    "HOSCPooling",
    "LaPooling",
    "JustBalancePooling",
    "KMISPooling",
    "MinCutPooling",
    "NDPPooling",
    "NMFPooling",
    "PANPooling",
    "SAGPooling",
    "TopkPooling",
]

pooler_map = {
    "asap": ASAPooling,
    "acc": AsymCheegerCutPooling,
    "diff": DiffPool,
    "dmon": DMoNPooling,
    "ec": EdgeContractionPooling,
    "graclus": GraclusPooling,
    "hosc": HOSCPooling,
    "lap": LaPooling,
    "jb": JustBalancePooling,
    "kmis": KMISPooling,
    "mincut": MinCutPooling,
    "ndp": NDPPooling,
    "nmf": NMFPooling,
    "pan": PANPooling,
    "sag": SAGPooling,
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
