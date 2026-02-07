from .base_conn import Connect, SparseConnect, sparse_connect
from .dense_conn import DenseConnect
from .eigenpool_conn import EigenPoolConnect
from .kron_conn import KronConnect

connect_functions = ["sparse_connect"]

connect_classes = [
    "Connect",
    "SparseConnect",
    "DenseConnect",
    "EigenPoolConnect",
    "KronConnect",
]

__all__ = connect_classes + connect_functions
