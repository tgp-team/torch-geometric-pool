from .base_conn import Connect, SparseConnect, sparse_connect
from .dense_conn import DenseConnect, dense_connect, postprocess_adj_pool
from .dense_conn_spt import DenseConnectSPT
from .kron_conn import KronConnect

connect_functions = [
    "sparse_connect",
    "dense_connect",
    "postprocess_adj_pool",
]

connect_classes = [
    "Connect",
    "SparseConnect",
    "DenseConnect",
    "KronConnect",
    "DenseConnectSPT",
]

__all__ = connect_classes + connect_functions
