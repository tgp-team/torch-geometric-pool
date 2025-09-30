from .base_reduce import BaseReduce, Reduce
from .global_reduce import dense_global_reduce, global_reduce
from .identity_reduce import IdentityReduce

reduce_functions = [
    "global_reduce",
    "dense_global_reduce",
]

reduce_classes = [
    "Reduce",
    "BaseReduce",
    "IdentityReduce",
]

__all__ = reduce_classes + reduce_functions
