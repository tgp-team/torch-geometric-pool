"""Common typing aliases used across :mod:`tgp`."""

from typing import Literal

# Strategy to build/obtain the inverse assignment matrix.
SinvType = Literal["transpose", "inverse"]

# Reduction alias names are backend-defined (scatter/PyG aggregations).
ReduceType = str

# Strategy used by Lift modules to map pooled features back to node space.
LiftType = Literal["transpose", "inverse", "precomputed"]

# Edge aggregation strategy used by sparse connectivity routines.
ConnectionType = Literal["sum", "mean", "min", "max", "mul"]
