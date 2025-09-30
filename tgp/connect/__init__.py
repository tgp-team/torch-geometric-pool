from .base_conn import Connect, SparseConnect, sparse_connect
from .dense_conn import DenseConnect
from .dense_conn_spt import DenseConnectSPT
from .identity_connect import IdentityConnect
from .kron_conn import KronConnect

connect_functions = ["sparse_connect"]

connect_classes = [
    "Connect",
    "SparseConnect",
    "DenseConnect",
    "IdentityConnect",
    "KronConnect",
    "DenseConnectSPT",
]

__all__ = connect_classes + connect_functions
