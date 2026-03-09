Typing
======
 
.. currentmodule:: tgp.utils.typing

Type aliases
------------

.. py:class:: SinvType

   Strategy used to compute :math:`\mathbf{S}_{\text{inv}}` from the
   assignment matrix :math:`\mathbf{S}`.
   Backed by ``typing.Literal["transpose", "inverse"]``.

.. py:class:: ReduceType

   Reduction alias forwarded to backend reducers/aggregators.
   Supported values depend on the backend context (e.g., scatter reductions
   or PyG aggregation aliases). Backed by :obj:`str`.

.. py:class:: LiftType

   Strategy used by lift operators when mapping pooled features back to
   original nodes.
   Backed by ``typing.Literal["transpose", "inverse", "precomputed"]``.

.. py:class:: ConnectionType

   Edge aggregation mode used when constructing pooled connectivity.
   Backed by ``typing.Literal["sum", "mean", "min", "max", "mul"]``.
