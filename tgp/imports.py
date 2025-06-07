try:
    import torch_cluster

    HAS_TORCH_CLUSTER = True
except ImportError:
    torch_cluster = None
    HAS_TORCH_CLUSTER = False

try:
    import torch_sparse
    from torch_sparse import SparseTensor

    HAS_TORCH_SPARSE = True
except ImportError:
    torch_sparse = None
    SparseTensor = "SparseTensor"
    HAS_TORCH_SPARSE = False

try:
    import torch_scatter

    HAS_TORCH_SCATTER = True
except ImportError:
    torch_scatter = None
    HAS_TORCH_SCATTER = False

try:
    import pygsp

    HAS_PYGSP = True
except ImportError:
    pygsp = None
    HAS_PYGSP = False


def check_torch_cluster_available():
    if not HAS_TORCH_CLUSTER:
        raise ImportError(
            "The 'torch_cluster' package is required for this operation. "
            "Please install it with `pip install torch-cluster`."
        )


def check_torch_sparse_available():
    if not HAS_TORCH_SPARSE:
        raise ImportError(
            "The 'torch_sparse' package is required for this operation. "
            "Please install it with `pip install torch-sparse`."
        )


def check_pygsp_available():
    if not HAS_PYGSP:
        raise ImportError(
            "The 'pygsp' package is required for this operation. "
            "Please install it with `pip install pygsp`."
        )
