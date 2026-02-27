.. py:module:: tgp.reduce
.. currentmodule:: tgp.reduce

Reduce
======

The most general interface is :class:`~tgp.reduce.Reduce`, which is the
parent class for every :math:`\texttt{reduce}` operator in :mod:`tgp`.

For graph-level aggregation, :class:`~tgp.reduce.AggrReduce` and
:func:`~tgp.reduce.readout` rely on PyG's
:class:`torch_geometric.nn.aggr.Aggregation` modules. The helper
function :func:`~tgp.reduce.get_aggr` lets you obtain these aggregators
by **string alias**, mirroring how :func:`tgp.poolers.get_pooler`
instantiates poolers by alias.

.. code-block:: python

   from tgp.reduce import get_aggr, AggrReduce, readout

   # Aggregator instance from alias
   aggr = get_aggr("mean")
   reducer = AggrReduce(aggr)

   # Use directly in readout
   x_graph = readout(x, reduce_op="set2set", in_channels=64, processing_steps=3)

Aggregator aliases
------------------

The table below lists the supported aliases for :func:`get_aggr` and
the corresponding PyG aggregation classes:

.. list-table::
   :header-rows: 1

   * - Alias
     - PyG aggregation class
   * - ``sum``
     - :class:`torch_geometric.nn.aggr.SumAggregation`
   * - ``mean``
     - :class:`torch_geometric.nn.aggr.MeanAggregation`
   * - ``max``
     - :class:`torch_geometric.nn.aggr.MaxAggregation`
   * - ``min``
     - :class:`torch_geometric.nn.aggr.MinAggregation`
   * - ``mul``
     - :class:`torch_geometric.nn.aggr.MulAggregation`
   * - ``var``
     - :class:`torch_geometric.nn.aggr.VarAggregation`
   * - ``std``
     - :class:`torch_geometric.nn.aggr.StdAggregation`
   * - ``softmax``
     - :class:`torch_geometric.nn.aggr.SoftmaxAggregation`
   * - ``power_mean``
     - :class:`torch_geometric.nn.aggr.PowerMeanAggregation`
   * - ``median``
     - :class:`torch_geometric.nn.aggr.MedianAggregation`
   * - ``quantile``
     - :class:`torch_geometric.nn.aggr.QuantileAggregation`
   * - ``lstm``
     - :class:`torch_geometric.nn.aggr.LSTMAggregation`
   * - ``gru``
     - :class:`torch_geometric.nn.aggr.GRUAggregation`
   * - ``set2set``
     - :class:`torch_geometric.nn.aggr.Set2Set`
   * - ``degree_scaler``
     - :class:`torch_geometric.nn.aggr.DegreeScalerAggregation`
   * - ``sort``
     - :class:`torch_geometric.nn.aggr.SortAggregation`
   * - ``multi``
     - :class:`torch_geometric.nn.aggr.MultiAggregation`
   * - ``attentional``
     - :class:`torch_geometric.nn.aggr.AttentionalAggregation`
   * - ``equilibrium``
     - :class:`torch_geometric.nn.aggr.EquilibriumAggregation`
   * - ``mlp``
     - :class:`torch_geometric.nn.aggr.MLPAggregation`
   * - ``deep_sets``
     - :class:`torch_geometric.nn.aggr.DeepSetsAggregation`
   * - ``set_transformer``
     - :class:`torch_geometric.nn.aggr.SetTransformerAggregation`
   * - ``lcm``
     - :class:`torch_geometric.nn.aggr.LCMAggregation`
   * - ``variance_preserving``
     - :class:`torch_geometric.nn.aggr.VariancePreservingAggregation`
   * - ``patch_transformer``
     - :class:`torch_geometric.nn.aggr.PatchTransformerAggregation`
   * - ``graph_multiset_transformer``
     - :class:`torch_geometric.nn.aggr.GraphMultisetTransformer`


.. autosummary::
    :nosignatures:
    {% for cls in tgp.reduce.reduce_classes %}
        {{ cls }}
    {% endfor %}

{% for cls in tgp.reduce.reduce_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}

{% for func in tgp.reduce.reduce_functions %}
.. autodata:: {{ func }}
   :annotation:
{% endfor %}