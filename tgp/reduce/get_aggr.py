"""Resolve string aliases to PyG Aggregation instances.

Use :func:`get_aggr` to obtain aggregators by name for :class:`AggrReduce` and
:func:`readout`. For parametrized aggregators (e.g. LSTM, Set2Set), pass
keyword arguments such as :obj:`in_channels`, :obj:`out_channels`,
:obj:`processing_steps`.
"""

from typing import Any

try:
    from torch_geometric.nn import aggr as _aggr_module
except Exception:
    _aggr_module = None

# Alias -> (PyG class name, default kwargs for parametrized aggrs)
_AGGR_ALIASES = {
    "sum": ("SumAggregation", {}),
    "mean": ("MeanAggregation", {}),
    "max": ("MaxAggregation", {}),
    "min": ("MinAggregation", {}),
    "mul": ("MulAggregation", {}),
    "var": ("VarAggregation", {}),
    "std": ("StdAggregation", {}),
    "softmax": ("SoftmaxAggregation", {}),
    "power_mean": ("PowerMeanAggregation", {}),
    "median": ("MedianAggregation", {}),
    "quantile": ("QuantileAggregation", {}),
    "lstm": ("LSTMAggregation", {}),
    "gru": ("GRUAggregation", {}),
    "set2set": ("Set2Set", {}),
    "degree_scaler": ("DegreeScalerAggregation", {}),
    "sort": ("SortAggregation", {}),
    "multi": ("MultiAggregation", {}),
    "attentional": ("AttentionalAggregation", {}),
    "equilibrium": ("EquilibriumAggregation", {}),
    "mlp": ("MLPAggregation", {}),
    "deep_sets": ("DeepSetsAggregation", {}),
    "set_transformer": ("SetTransformerAggregation", {}),
    "lcm": ("LCMAggregation", {}),
    "variance_preserving": ("VariancePreservingAggregation", {}),
    "patch_transformer": ("PatchTransformerAggregation", {}),
    "graph_multiset_transformer": ("GraphMultisetTransformer", {}),
}


def get_aggr(alias: str, **kwargs: Any) -> Any:
    r"""Return a PyG :class:`torch_geometric.nn.aggr.Aggregation` instance by alias.

    Use this with :class:`~tgp.reduce.AggrReduce` or :func:`~tgp.reduce.readout`
    when you want to specify the aggregator by string instead of passing a
    module instance.

    Args:
        alias: Name of the aggregator (e.g. :obj:`"sum"`, :obj:`"mean"`,
            :obj:`"lstm"`, :obj:`"set2set"`). Case-insensitive.
        **kwargs: Passed to the aggregator constructor. Parametrized aggregators
            typically need :obj:`in_channels`, :obj:`out_channels`, and/or
            :obj:`processing_steps` (e.g. for Set2Set, LSTM).

    Returns:
        An instance of the requested PyG Aggregation.

    Raises:
        ImportError: If :obj:`torch_geometric.nn.aggr` is not available.
        ValueError: If :obj:`alias` is not recognized.

    Example:
        >>> from tgp.reduce import get_aggr, AggrReduce
        >>> red = AggrReduce(get_aggr("mean"))
        >>> red_lstm = AggrReduce(get_aggr("lstm", in_channels=64, out_channels=64))
    """
    if _aggr_module is None:
        raise ImportError(
            "get_aggr requires torch_geometric.nn.aggr. "
            "Install PyTorch Geometric to use PyG aggregations."
        )
    key = alias.strip().lower().replace("-", "_")
    if key not in _AGGR_ALIASES:
        raise ValueError(
            f"Unknown aggregator alias: {alias!r}. "
            f"Known aliases: {sorted(_AGGR_ALIASES.keys())}"
        )
    class_name, default_kw = _AGGR_ALIASES[key]
    cls = getattr(_aggr_module, class_name, None)
    if cls is None:
        raise ValueError(
            f"Aggregator {class_name!r} not found in torch_geometric.nn.aggr. "
            "Your PyG version may not include it."
        )
    merged = {**default_kw, **kwargs}
    # Some PyG aggrs use positional in_channels, out_channels
    if (
        key in ("lstm", "gru")
        and "out_channels" not in merged
        and "in_channels" in merged
    ):
        merged["out_channels"] = merged["in_channels"]
    return cls(**merged)
