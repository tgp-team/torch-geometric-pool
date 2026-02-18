from .aggr_reduce import AggrReduce
from .base_reduce import BaseReduce, Reduce
from .eigenpool_reduce import EigenPoolReduce
from .global_reduce import readout

reduce_functions = [
    "readout",
]

reduce_classes = [
    "AggrReduce",
    "Reduce",
    "BaseReduce",
    "EigenPoolReduce",
]

__all__ = reduce_classes + reduce_functions
