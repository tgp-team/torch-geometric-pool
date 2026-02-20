from .aggr_reduce import AggrReduce
from .base_reduce import BaseReduce, Reduce
from .eigenpool_reduce import EigenPoolReduce
from .get_aggr import get_aggr
from .global_reduce import readout

reduce_functions = [
    "get_aggr",
    "readout",
]

reduce_classes = [
    "AggrReduce",
    "Reduce",
    "BaseReduce",
    "EigenPoolReduce",
]

__all__ = reduce_classes + reduce_functions
