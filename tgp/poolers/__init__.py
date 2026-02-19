import inspect

from .asap import ASAPooling
from .asym_cheeger_cut import AsymCheegerCutPooling
from .bnpool import BNPool
from .diffpool import DiffPool
from .dmon import DMoNPooling
from .edge_contraction import EdgeContractionPooling
from .eigenpool import EigenPooling
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
from .sep import SEPPooling
from .topk import TopkPooling

pooler_classes = [
    "ASAPooling",
    "AsymCheegerCutPooling",
    "BNPool",
    "DiffPool",
    "DMoNPooling",
    "EdgeContractionPooling",
    "EigenPooling",
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
    "SEPPooling",
    "TopkPooling",
]

pooler_map = {
    "asap": ASAPooling,
    "acc": AsymCheegerCutPooling,
    "bnpool": BNPool,
    "diff": DiffPool,
    "dmon": DMoNPooling,
    "ec": EdgeContractionPooling,
    "eigen": EigenPooling,
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
    "sep": SEPPooling,
    "topk": TopkPooling,
}


def _missing_required_init_kwargs(pooler_cls, provided_kwargs: dict) -> list[str]:
    """Return required ``__init__`` kwargs not present in ``provided_kwargs``."""
    missing = []
    init_sig = inspect.signature(pooler_cls.__init__)
    for name, param in init_sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty and name not in provided_kwargs:
            missing.append(name)
    return missing


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
    if pooler_name.endswith("_u"):
        base_name = pooler_name[:-2]
        if base_name not in pooler_map:
            raise ValueError(
                f"Unknown pooler_name='{pooler_name}'. "
                f"Available poolers: {list(pooler_map.keys())}"
            )
        pooler_name = base_name
        kwargs.setdefault("batched", False)

    if pooler_name not in pooler_map:
        raise ValueError(
            f"Unknown pooler_name='{pooler_name}'. "
            f"Available poolers: {list(pooler_map.keys())}"
        )

    pooler_cls = pooler_map[pooler_name]
    signature = pooler_cls.get_signature()

    if signature.has_kwargs:
        init_kwargs = kwargs
    else:
        # Filter out any kwargs that aren't in the signature:
        init_kwargs = {k: v for k, v in kwargs.items() if k in signature.args}

    missing_required = _missing_required_init_kwargs(pooler_cls, init_kwargs)
    if missing_required:
        required = ", ".join(missing_required)
        raise TypeError(
            f"Missing required argument(s) for pooler '{pooler_name}' "
            f"({pooler_cls.__name__}): {required}"
        )

    try:
        return pooler_cls(**init_kwargs)
    except TypeError as exc:
        # Re-check after constructor call in case dynamic signatures differ.
        missing_required = _missing_required_init_kwargs(pooler_cls, init_kwargs)
        if missing_required:
            required = ", ".join(missing_required)
            raise TypeError(
                f"Missing required argument(s) for pooler '{pooler_name}' "
                f"({pooler_cls.__name__}): {required}"
            ) from exc
        raise
