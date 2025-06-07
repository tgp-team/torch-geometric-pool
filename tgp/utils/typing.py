from typing import Literal

SinvType = Literal["transpose", "inverse"]
ReduceType = Literal["sum", "mean", "min", "max", "any"]
LiftType = Literal["transpose", "inverse", "precomputed"]
ConnectionType = Literal["sum", "mean", "min", "max", "mul"]
