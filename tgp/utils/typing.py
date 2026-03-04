from typing import Literal

SinvType = Literal["transpose", "inverse"]
ReduceType = (
    str  # Reduction names are backend-defined (PyG aggregations for readout/reduce ops)
)
LiftType = Literal["transpose", "inverse", "precomputed"]
ConnectionType = Literal["sum", "mean", "min", "max", "mul"]
