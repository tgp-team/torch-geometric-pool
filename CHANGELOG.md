## v1.0.1

- **HOSCPooling higher-order cut loss is now sparse**  
  Replaces the dense motif-adjacency path (`A^3` via `to_dense_adj`) with
  `sparse_ho_mincut_loss`, avoiding dense \((N, N)\) materialization and adding
  coverage for empty-edge and mixed-edge batch cases.
- **Connectivity helpers reject dense adjacencies**  
  `connectivity_to_edge_index`, `connectivity_to_torch_coo`, and
  `connectivity_to_sparsetensor` now raise clear `ValueError`s when given dense
  adjacency matrices \((N, N)\) / \((B, N, N)\), with tests added.
- **Minor fixes**

## v1.0
This release summarizes the recent architectural refactor and behavioral updates.
It focuses on public‑facing API/behavior, intended usage, and design trade‑offs.

### Highlights
- **Unified dense pooling & connectivity**  
  Dense poolers share a common `DenseSRCPooling` base, accept raw sparse inputs, and
  use a single `sparse_output` flag to choose between block‑diagonal sparse and
  batched dense pooled adjacencies. `DenseConnect` now handles both batched dense
  inputs and unbatched sparse inputs in one place.
- **Standalone readout**  
  Graph‑level readout is handled by `GlobalReduce` / `AggrReduce`, decoupled from
  individual poolers and with strict shape/mask validation.
- **Expanded pooler family**  
  New or extended methods include:
  - `EigenPooling` (with `EigenPoolSelect`, `EigenPoolReduce`, `EigenPoolConnect`,
    `EigenPoolLift`);
  - `SEPPooling` / `SEPSelect` (structural‑entropy–guided, non‑trainable pooling);
  - `NMFPooling(batched=False)` and generalized `NMFSelect` for dense + sparse inputs;
  - `_u` suffix convention for unbatched dense variants (e.g. `bnpool_u`,
    `diff_u`, `dmon_u`, `acc_u`, `hosc_u`, `jb_u`).
- **Multi‑level precoarsening**  
  `PreCoarsening` supports:
  - a single pooler/config or a sequence of per‑level configs (pooler instance,
    alias string, `(name, kwargs)` tuples, or dicts);
  - repeated levels like `["sep", "sep", "sep"]` or `["ndp", "ndp", "ndp"]`;
  - multi‑level SEP hierarchies via `SEPSelect.multi_level_select`;
  - efficient “collapsed runs” internally while still exposing one `pooled_data`
    entry per requested level.
- **Stability & diagnostics**  
  - Spectral / BCE / balance losses handle empty‑graph and corner cases without NaNs;
  - `get_random_map_mask` behaves correctly when some graphs have zero kept nodes;
  - balance / orthogonality losses respect masks in batched settings;
  - dense poolers’ `extra_repr` consistently reports `batched` and `sparse_output`;
  - `tgp.utils.ops` functions are documented and exported.
---
### Dense pooling, masks and readout
#### Modes and `sparse_output`
Dense poolers implement **two internal processing modes**:
1. **Batched mode (`batched=True`)**
   - Converts sparse inputs to dense padded tensors:
     - `X`: `[B, Nmax, F]`
     - `A`: `[B, Nmax, Nmax]`
     - `S`: `[B, Nmax, K]`
   - **Pros**: vectorized operations, typically faster.
   - **Cons**: higher memory due to padding and densifying adjacency.
2. **Unbatched mode (`batched=False`)**
   - Avoids padding and operates on unbatched dense assignments:
     - `X`: `[N, F]`
     - `A`: sparse connectivity
     - `S`: `[N, K]`
   - **Pros**: memory‑efficient (no padding, no materializing non‑edges).
   - **Cons**: slower (per‑graph loops / less vectorization).
`sparse_output` controls the **format** of the pooled adjacency:
- `sparse_output=True`: block‑diagonal sparse output
  (`edge_index`, `edge_weight`, `batch`);
- `sparse_output=False`: batched dense adjacency of shape `[B, K, K]`.
This flag determines the appropriate downstream MP / global pooling layers.
#### Masks and supernode validity
- External `mask` arguments are honored **only** when inputs are already
  dense/padded; for sparse inputs, masks are created during preprocessing.
- **`SelectOutput.in_mask`**
  - optional stored attribute (`None` by default);
  - batched‑only with shape `[B, N]`;
  - mask on **original nodes**; used by `is_expressive`.
- **`SelectOutput.out_mask`**
  - derived property (not stored);
  - for `s` with shape `[B, N, K]`, output shape is `[B, K]`;
  - for `s` with shape `[N, K]`, output is `[B, K]` when `batch` is present,
    else `[1, K]`;
  - returns `None` for sparse assignments.
- **`PoolingOutput.mask`**
  - property that returns `so.out_mask` when `so` is set, else `None`;
  - there is no stored `mask` field;
  - downstream dense MP / global pool layers use it to ignore padded supernodes.
#### Readout (`GlobalReduce` / `AggrReduce`)
- **`GlobalReduce`**
  - import from `tgp.reduce`:
    `from tgp.reduce import GlobalReduce`;
  - infers sparse vs dense from `x`:
    - 2D `[N, F]` → sparse readout (use `batch` for grouping),
    - 3D `[B, N, F]` → dense readout;
  - node dimension is fixed (second‑to‑last): no `node_dim` argument.
- Reduce ops:
  - PyG‑style aggregators (string alias or
    `torch_geometric.nn.aggr.Aggregation` instance);
  - `"any"` is not supported; use `"sum"`, `"mean"`, `"max"`, `"min"`,
    `"lstm"`, `"set2set"`, etc., via `GlobalReduce(reduce_op, **kwargs)` or
    `get_aggr(reduce_op, **kwargs)` with `AggrReduce`.
- Mask validation:
  - for sparse (2D) `x`, passing `mask` raises `ValueError`;
  - for dense (3D) `x`, `mask` must be `[B, N]` or a `ValueError` is raised.
- Size semantics for sparse readout:
  - `size` is only valid when `batch` is provided;
  - with `batch=None`, omit `size`; `GlobalReduce` returns `[1, F]`.
Poolers no longer expose a `readout` method; always call `GlobalReduce`
(or `AggrReduce` directly) on the pooled node features.
---
### Reduce and utility changes
- **`BaseReduce`**
  - always computes :math:`S^T X`:
    - sparse assignments use scatter (optionally weighted by
      `so.s.values()` / `so.weight`);
    - dense assignments use matrix multiplication;
  - no longer accepts `reduce_op`; for mean / max / min etc. use `AggrReduce`
    with a PyG aggregator.
  - for dense `[N, K]` + `batch` (multi‑graph batches) it still unbatches per
    graph for memory efficiency when using `batched=False`.
- **`AggrReduce`**
  - wraps a PyG `Aggregation`;
  - supports `so=None` for graph‑level readout (one pooled node per graph);
  - dense `SelectOutput` assignments are intentionally unsupported; use
    `BaseReduce` for dense/soft reductions.
- **`get_aggr(alias, **kwargs)` in `tgp.reduce`**
  - resolves string aliases like `"sum"`, `"mean"`, `"lstm"`, `"set2set"` to
    concrete PyG aggregation classes;
  - merges defaults with user kwargs, filtering kwargs to what the PyG class
    actually accepts.
- **`tgp.utils.ops` post‑processing and helpers**
  - `postprocess_adj_pool_dense(...)` and `postprocess_adj_pool_sparse(...)`
    centralize:
    - self‑loop removal,
    - symmetric degree normalization,
    - optional edge‑weight normalization;
  - `dense_to_block_diag` converts dense `[B, K, K]` pooled adjacencies into
    block‑diagonal `edge_index` + `edge_weight`;
  - `is_dense_adj` detects dense adjacency tensors.
---
### API & behavior changes
- **`sparse_output` replaces legacy flags**  
  Flags such as `block_diags_output` and `unbatched_output` are removed in favor
  of `sparse_output` throughout the codebase.
- **`DenseConnect` behavior**  
  - batched dense path: accepts dense adjacencies `[B, N, N]` and dense
    assignments `[B, N, K]`, returns `[B, K, K]`;
  - unbatched sparse path: accepts sparse connectivity plus dense assignments
    `[N, K]`, returns dense `[B, K, K]` or block‑diagonal sparse connectivity
    depending on `sparse_output`.
- **`DenseSelect` → `MLPSelect`**  
  The dense selector has been renamed and moved from `dense_select.py` to
  `mlp_select.py`.
- **`MLPSelect` / `DPSelect`**  
  - accept `batched_representation` to choose between batched `[B, N, K]`
    and unbatched `[N, K]` assignments;
  - add `batch` to `SelectOutput` when operating unbatched.
- **Removed legacy classes/files**
  - `SparseBNPool` has been removed; use `BNPool(batched=False)` or
    `get_pooler("bnpool_u")`;
  - `dense_conn_spt.py` has been removed; dense connectivity logic lives in
    `dense_conn.py`.
- **`get_pooler` `_u` suffix**  
  Dense poolers implementing unbatched dense modes can be instantiated as
  `"<name>_u"` (e.g. `bnpool_u`, `diff_u`, `dmon_u`, `acc_u`, `hosc_u`, `jb_u`,
  `lap_u`, `mincut_u`). Unsupported `_u` names raise an error.
---
### Pooler coverage (batched / unbatched)
All dense poolers described below share the same output‑format semantics:
when they support both dense and block‑diagonal sparse outputs, the choice
is controlled by `sparse_output` as described above.
- **BNPool**
  - unified dense/sparse behavior; batched/unbatched branches share a consistent
    interface;
  - unbatched mode uses sparse losses and supports `num_neg_samples` to cap
    negative sampling.
- **LaPooling**
  - default: `batched=True`;
  - unbatched mode is kept for memory efficiency and computes per‑graph
    similarities to limit memory.
- **MinCutPooling**
  - unbatched mode (`batched=False` / `get_pooler("mincut_u")`) uses sparse
    connectivity and sparse MinCUT / orthogonality losses.
- **DiffPool**
  - unbatched mode (`batched=False` / `get_pooler("diff_u")`) uses sparse
    connectivity and sparse link‑prediction / entropy losses.
- **JustBalancePooling (JBPool)**
  - unbatched mode (`batched=False` / `get_pooler("jb_u")`) uses sparse
    connectivity and balance loss.
- **DMoNPooling**
  - unbatched mode (`batched=False` / `get_pooler("dmon_u")`) uses sparse
    connectivity and sparse spectral / cluster / orthogonality losses.
- **HOSCPooling**
  - unbatched mode (`batched=False` / `get_pooler("hosc_u")`) uses sparse
    connectivity and HOSC/orthogonality losses.
- **AsymCheegerCutPooling (ACC)**
  - unbatched mode (`batched=False` / `get_pooler("acc_u")`) uses sparse
    connectivity and sparse total‑variation / balance losses.
- **NMFPooling / `NMFSelect`**
  - `NMFPooling` supports both batched and unbatched modes, operating on sparse
    connectivity without padding in the unbatched case;
  - `NMFSelect` accepts both dense and sparse connectivity inputs, supports
    single‑graph and multi‑graph sparse batches, and falls back to trivial
    assignments for tiny graphs / `k > N`.
- **EigenPooling / `EigenPoolSelect`**
  - `EigenPooling` provides spectral‑clustering‑based pooling with
    eigenvector‑based reduce and lift (`EigenPoolReduce`, `EigenPoolLift`);
  - `EigenPoolSelect` runs spectral clustering per graph and computes an
    eigenvector pooling matrix `theta`, supporting both single‑graph and
    batched sparse inputs.
- **SEPPooling / `SEPSelect`**
  - SEP is non‑trainable and topology‑driven, building a coding tree and
    exposing depth‑1 partitions in the standard forward pass;
  - `SEPSelect.multi_level_select` returns multi‑level assignments used by
    `SEPPooling.multi_level_precoarsening` and `PreCoarsening` with repeated
    `"sep"` levels.
---
### Bug fixes
- **Spectral loss** (batched and unbatched) no longer returns NaN for graphs
  with zero edges; empty graphs now contribute zero loss and the result remains
  finite.
- **Weighted BCE reconstruction loss** now counts edges using a boolean mask to
  avoid mismatches when adjacency is non‑binary or has clamped edges.
- **Batched random assignment** in `get_random_map_mask` now handles graphs with
  zero kept nodes by falling back to global assignments.
- **AsymNorm and HOSC orthogonality losses** now handle single‑node /
  single‑cluster edge cases without NaNs or index errors, returning finite,
  well‑defined values.
- **JustBalance (JB) loss** in batched mode now correctly applies per‑graph
  normalization when a node mask is provided (e.g. variable‑sized graphs with
  zero padding).
- **AsymNorm (ACC) loss** in batched mode now computes correctly per graph when a
  node mask is provided: the batched path builds a flat assignment and batch
  vector from the mask and delegates to the unbatched implementation.
---
### Migration notes
- **Readout**  
  Replace `global_reduce(x, ...)`, `dense_global_reduce(x, ...)` and
  pooler‑specific `readout(x, ...)` helpers with `GlobalReduce` modules (e.g.
  `self.readout = GlobalReduce("sum")` in `__init__` and
  `x = self.readout(x, batch=..., mask=...)` in `forward`). Replace
  `pooler.global_pool(x, ...)` with calls to a `GlobalReduce` instance.
  Remove `node_dim` from pooler constructors and from KMIS if you used it.
- **Output format flags**  
  If you previously relied on `block_diags_output` or `unbatched_output`,
  update to `sparse_output`.
- **Unbatched dense modes**  
  Use `_u` pooler names (e.g. `bnpool_u`, `lap_u`, `diff_u`, `dmon_u`, `acc_u`,
  `hosc_u`, `jb_u`, `mincut_u`) to select unbatched dense modes.
- **Dense preprocessing**  
  Dense poolers should no longer require explicit calls to `preprocessing(...)`
  in user code; they perform it internally when `batched=True`.
- **PreCoarsening configuration**  
  `PreCoarsening` now takes a single `poolers` argument (one pooler/config or a
  sequence for multilevel). The recursive depth argument is no longer supported:
  the number of levels is given by `len(poolers)`, e.g. use
  `poolers=["ndp", "ndp", "ndp"]` for three NDPPooling levels.
---

## v0.4 - Internal Sparse Format Migration

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
