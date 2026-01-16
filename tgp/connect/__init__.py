from .base_conn import Connect, SparseConnect, sparse_connect
from .dense_conn import DenseConnect
from .dense_conn_spt import DenseConnectUnbatched
from .kron_conn import KronConnect

connect_functions = ["sparse_connect"]

connect_classes = [
    "Connect",
    "SparseConnect",
    "DenseConnect",
    "KronConnect",
    "DenseConnectUnbatched",
]

__all__ = connect_classes + connect_functions
