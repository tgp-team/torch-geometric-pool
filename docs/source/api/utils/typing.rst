Typing
======
 
.. currentmodule:: tgp.utils.typing

Type aliases
------------

.. py:data:: SinvType
   :type: typing.Literal["transpose", "inverse"]

   Strategy used to compute :math:`\mathbf{S}_{\text{inv}}` from the
   assignment matrix :math:`\mathbf{S}`.

.. py:data:: ReduceType
   :type: str

   Reduction alias forwarded to backend reducers/aggregators.
   Supported values depend on the backend context (e.g., scatter reductions
   or PyG aggregation aliases).

.. py:data:: LiftType
   :type: typing.Literal["transpose", "inverse", "precomputed"]

   Strategy used by lift operators when mapping pooled features back to
   original nodes.

.. py:data:: ConnectionType
   :type: typing.Literal["sum", "mean", "min", "max", "mul"]

   Edge aggregation mode used when constructing pooled connectivity.
