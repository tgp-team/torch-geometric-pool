# Torch Geometric Pool — Changelog (Internal Sparse Format Migration Edition)

### Overview

This release migrates tgp's internal sparse tensor representation from `torch_sparse.SparseTensor` to PyTorch's native `torch.sparse_coo_tensor`. This change **eliminates the internal dependency** on `torch_sparse` while maintaining full backward compatibility for users who work with `SparseTensor` inputs/outputs.

**Key principle:** Internally, tgp now uses `torch.sparse_coo_tensor` exclusively. However, if users provide `SparseTensor` data as input, the library still accepts it and can return results in the same format.

---

### Breaking Changes

#### `SelectOutput.s` Internal Representation

The assignment matrix `s` in `SelectOutput` is now stored as a `torch.sparse_coo_tensor` instead of `torch_sparse.SparseTensor`.

**Type annotation change:**
```python
# Old (tgp_dev)
s: Union[SparseTensor, Tensor]

# New (tgp)
s: Tensor  # Can be dense or torch.sparse_coo_tensor
```

**If you access sparse storage directly, update your code:**

| Old API (SparseTensor)           | New API (torch COO)                       |
|----------------------------------|-------------------------------------------|
| `so.s.storage.row()`             | `so.s.indices()[0]` or `so.node_index`    |
| `so.s.storage.col()`             | `so.s.indices()[1]` or `so.cluster_index` |
| `so.s.storage.value()`           | `so.s.values()` or `so.weight`            |
| `so.s.sparse_sizes()`            | `so.s.size()`                             |
| `so.s.nnz()`                     | `so.s._nnz()`                             |
| `isinstance(so.s, SparseTensor)` | `so.s.is_sparse` or `so.is_sparse`        |

**Note:** The `SelectOutput` properties `node_index`, `cluster_index`, and `weight` abstract these differences, so prefer using them when possible.

---

### New Features

#### `is_sparsetensor()` Helper Function

New utility function in `tgp/imports.py` to detect `torch_sparse.SparseTensor` objects:

```python
from tgp.imports import is_sparsetensor

if is_sparsetensor(adj):
    # Handle SparseTensor input
    ...
```

#### `connectivity_to_torch_coo()` Conversion Function

New function in `tgp/utils/ops.py` to convert any adjacency format to `torch.sparse_coo_tensor`:

```python
from tgp.utils import connectivity_to_torch_coo

# Converts edge_index, SparseTensor, or existing torch COO to torch COO format
adj_coo = connectivity_to_torch_coo(edge_index, edge_weight, num_nodes)
```

#### `propagate_assignments_sparse()` - Memory-Efficient Assignment Propagation

New function in `tgp/utils/ops.py` for fully sparse assignment propagation:

```python
from tgp.utils.ops import propagate_assignments_sparse

# O(E) memory instead of O(N*K) - no dense [num_nodes, num_kept] tensors created
assignments, mapping, mask = propagate_assignments_sparse(
    assignments, edge_index, kept_node_tensor, mask, num_clusters
)
```

---

### Changes by Module

#### `tgp/select/base_select.py`

**`SelectOutput` class:**
- Type of `s` changed from `Union[SparseTensor, Tensor]` to `Tensor`
- `is_sparse` property now checks `self.s.is_sparse` instead of `isinstance(self.s, SparseTensor)`
- Properties `node_index`, `cluster_index`, `weight` now use `s.indices()` and `s.values()` instead of `s.storage.*`

**`cluster_to_s()` function:**
```python
# Old
return SparseTensor(row=node_index, col=cluster_index, value=weight, sparse_sizes=(N, K))

# New
indices = torch.stack([node_index, cluster_index], dim=0)
return torch.sparse_coo_tensor(indices=indices, values=values, size=(N, K), is_coalesced=True)
```

#### `tgp/utils/ops.py`

**Renamed function:**
- `connectivity_to_sparse_tensor()` → `connectivity_to_sparsetensor()` (clarifies it returns `torch_sparse.SparseTensor`)

**Updated `connectivity_to_edge_index()`:**
- Now handles `torch.sparse_coo_tensor` inputs via `.indices()` and `.values()`
- Clones tensors to avoid returning views that share memory with the sparse tensor

**Updated `pseudo_inverse()`:**
```python
# Old
adj_inv = SparseTensor.from_dense(adj_inv)

# New
adj_inv = adj_inv.to_sparse_coo()
```

**Updated `delta_gcn_matrix()`:**
```python
# Old - using torch_sparse utilities
eye_index, eye_weight = torch_sparse_eye(num_nodes, device=device, dtype=dtype)
propagation_matrix = SparseTensor(row=..., col=..., value=..., sparse_sizes=...).coalesce("sum")

# New - using native PyTorch
diag_indices = torch.arange(num_nodes, device=device)
eye_index = torch.stack([diag_indices, diag_indices], dim=0)
eye_weight = torch.ones(num_nodes, device=device, dtype=dtype)
propagation_matrix = torch.sparse_coo_tensor(combined_indices, combined_values, size=...).coalesce()
```

**Updated `get_assignments()`:**
- Uses `propagate_assignments_sparse()` for memory-efficient sparse operations
- Works with edge_index in `[2, E]` format or torch COO sparse tensors

#### `tgp/imports.py`

**Added:**
```python
def is_sparsetensor(obj):
    """Check if object is a torch_sparse.SparseTensor (if library is available)."""
    if not HAS_TORCH_SPARSE:
        return False
    return isinstance(obj, SparseTensor)
```

#### `tgp/reduce/base_reduce.py`

**Removed imports:**
- `import torch_sparse`
- `from torch_sparse import SparseTensor`

**Updated `BaseReduce.forward()` (sparse assignment path):**

*Note: The current implementation also handles dense assignments (`[B, N, K]` and unbatched `[N, K]` with multi-graph batches), but this changelog focuses on the SparseTensor migration.*

```python
# Old - using torch_sparse.matmul
if isinstance(so.s, SparseTensor):
    x_pool = torch_sparse.matmul(so.s.t(), x, reduce=self.operation)

# New - using scatter with indices from torch COO
if so.s.is_sparse:
    src = x[so.node_index]
    values = so.s.values()
    if values is not None and self.operation != "any":
        src = src * values.view(-1, 1)
    x_pool = scatter(src, so.cluster_index, dim=0, dim_size=so.num_supernodes, reduce=reduce)
```

**Updated `Reduce.reduce_batch()`:**
```python
# Old
assert isinstance(select_output.s, SparseTensor)

# New - handles both sparse and dense assignments
if select_output.s.is_sparse:
    out = torch.arange(select_output.num_supernodes, device=batch.device)
    return out.scatter_(0, select_output.cluster_index, batch[select_output.node_index])
else:
    # Dense [N, K] tensor: each graph in the batch has K supernodes
    K = select_output.num_supernodes
    batch_size = int(batch.max().item()) + 1
    return torch.arange(batch_size, device=batch.device).repeat_interleave(K)
```

#### `tgp/connect/base_conn.py`

**Updated `sparse_connect()`:**
- Added detection for torch COO sparse tensors: `isinstance(edge_index, Tensor) and edge_index.is_sparse`
- Added format preservation: tracks `to_torch_coo` flag and converts back using `connectivity_to_torch_coo()`

```python
# Old
to_sparse = isinstance(edge_index, SparseTensor)

# New
to_sparsetensor = is_sparsetensor(edge_index)
to_torch_coo = isinstance(edge_index, Tensor) and edge_index.is_sparse
```

**Updated imports:**
```python
# Old
from tgp_dev.utils import connectivity_to_sparse_tensor

# New
from tgp.utils import connectivity_to_sparsetensor, connectivity_to_torch_coo
```

### Backward Compatibility

The following mechanisms ensure users can still work with `torch_sparse.SparseTensor`:

1. **Input handling:** All public functions detect `SparseTensor` via `is_sparsetensor()` and convert internally to torch COO
2. **Output format preservation:** Functions track the input format and convert back before returning
3. **`connectivity_to_sparsetensor()`:** Dedicated function to convert any format to `SparseTensor` when needed

**Example flow:**
```python
# User provides SparseTensor input
edge_index = SparseTensor.from_edge_index(...)

# Internally, tgp converts to torch COO for processing
# ...

# Output is converted back to SparseTensor to match input format
pooled_adj, _ = pooler.connect(edge_index, so)  # Returns SparseTensor
```

---

### Benefits

1. **Reduced dependencies:** `torch_sparse` is no longer required for core functionality
3. **Native PyTorch integration:** Better compatibility with PyTorch ecosystem, autograd, and future optimizations
4. **Simplified codebase:** Uses consistent PyTorch sparse tensor APIs throughout
5. **Future-proof:** PyTorch's sparse tensor support continues to improve with each release

---

### Migration Guide

#### For users who don't access `SelectOutput.s` internals

**No changes required.** The public API remains the same.

#### For users who access sparse storage directly

Replace direct storage access with properties or native torch COO methods:

```python
# Old code
row_indices = so.s.storage.row()
col_indices = so.s.storage.col()
values = so.s.storage.value()
is_sparse = isinstance(so.s, SparseTensor)

# New code (option 1: use SelectOutput properties)
row_indices = so.node_index
col_indices = so.cluster_index
values = so.weight
is_sparse = so.is_sparse

# New code (option 2: use torch COO API directly)
row_indices = so.s.indices()[0]
col_indices = so.s.indices()[1]
values = so.s.values()
is_sparse = so.s.is_sparse
```

#### For users who check tensor types

```python
# Old code
from torch_sparse import SparseTensor
if isinstance(tensor, SparseTensor):
    ...

# New code
from tgp.imports import is_sparsetensor
if is_sparsetensor(tensor):
    # It's a torch_sparse.SparseTensor
    ...
elif isinstance(tensor, Tensor) and tensor.is_sparse:
    # It's a torch.sparse_coo_tensor
    ...
```
