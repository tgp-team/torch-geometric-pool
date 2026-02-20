# Torch Geometric Pool — Changelog (Refactor Edition)

This changelog summarizes the recent architectural refactor and behavioral updates.
It focuses on public‑facing API/behavior, intended usage, and design tradeoffs.

## Highlights

- **Unified dense pooling interface**: Dense poolers now accept raw sparse inputs and
  handle batching/densification internally when `batched=True`.
  
- **Explicit output format control**: A single flag, `sparse_output`, now determines
  whether pooled adjacency is returned as block‑diagonal sparse or batched dense.

- **Two masks (in_mask / out_mask):** `SelectOutput` has `in_mask` (stored, optional;
  mask on original nodes `[B, N]`) and `out_mask` (property; mask on pooled nodes
  `[B, K]`). `PoolingOutput.mask` is derived from `so.out_mask`. **Node-level masks
  are batched-only everywhere:** readout and `AggrReduce` accept `mask` only for
  dense (3D) x; `mask` must be `None` for unbatched (2D) x.

- **BNPool unified**: Dense and sparse BNPool variants are merged into one class with
  batched/unbatched branches and consistent outputs.

- **MinCut unbatched mode**: MinCutPooling now supports `batched=False` with sparse
  losses and optional sparse outputs.

- **DenseConnect consolidated**: All dense connect logic (batched and unbatched paths)
  lives in `DenseConnect`; redundant variants were removed.

- **EigenPooling implemented and integrated**:
  - Added `EigenPooling` end-to-end pipeline with dedicated
    `EigenPoolSelect`, `EigenPoolReduce`, `EigenPoolConnect`, and `EigenPoolLift`.
  - Exported EigenPool modules through package `__init__` files and enabled
    factory creation via `get_pooler("eigen")`.
  - Added sparse multi-graph support via `edge_index` + `batch`, including
    pre-coarsening support through the shared mixin.

- **SEP pooling added** (Wu et al., ICML 2022 — Structural Entropy Guided Graph
  Hierarchical Pooling):
  - `SEPPooling` and `SEPSelect` implement topology-driven clustering via a
    coding tree; single-level forward returns the depth-1 partition.
  - Non-trainable; integrated with `get_pooler("sep")` and the SRC pipeline.

- **NMF unbatched path implemented**:
  - `NMFPooling(batched=False)` is now supported.
  - Works with sparse connectivity in unbatched mode and supports both dense
    and sparse pooled adjacency outputs via `sparse_output`.

- **NMF select generalized**:
  - `NMFSelect` now accepts dense and sparse connectivity inputs.
  - Supports single-graph and multi-graph sparse batches.
  - Handles edge cases robustly (e.g., small graphs / `k > N`) with consistent
    output shaping and padding behavior.

- **PreCoarsening transform — multiple poolers and multilevel behavior**:
  - `PreCoarsening` takes a single `poolers` argument: either one pooler/config
    (one level) or a sequence of per-level configs (pooler instance, alias string,
    `(name, kwargs)` tuple, or dict). The same pooler can be repeated for multiple
    levels (e.g. `["sep", "sep", "sep"]`).
  - Supports multilevel precoarsening: building several coarsened graphs in one go
    (level 1 = first pooling, level 2 = pool again, etc.). For poolers like SEP
    that naturally produce a hierarchy (e.g. one coding tree with multiple levels),
    the transform uses that internal hierarchy instead of applying a single-level
    step repeatedly, so the result is a proper multilevel coarsening with one entry
    in `pooled_data` per level.
  - Enables mixed schedules (e.g. NDP then SEP then Graclus) and per-level options,
    all via the `poolers` sequence.

- **Dense pooler repr fixes**:
  - `extra_repr` output for dense poolers now consistently reports
    `batched` and `sparse_output`, improving debuggability and config visibility.

- **Ops in docs**: The :mod:`tgp.utils.ops` utilities are now included in the API
  documentation under Utils.

## Reduce cleanup and readout refactor

- **BaseReduce** now only computes :math:`S^T X`: for sparse assignment it is a sum
  (with optional weights from :obj:`so.s.values()`); for dense assignment it is a
  matrix multiply. It **no longer accepts** :obj:`reduce_op`. For aggregation types
  such as mean, max, or min, use **AggrReduce** with a PyG aggregator, e.g.
  :obj:`AggrReduce(get_aggr("mean"))` or :obj:`AggrReduce(MeanAggregation())`.
- **Unbatching in BaseReduce** for dense multi-graph (dense :math:`[N, K]` with a
  batch vector) is **unchanged**: each graph is processed separately for memory
  efficiency when using unbatched dense poolers (:obj:`batched=False`).
- **get_aggr(alias, **kwargs)**: New helper in :obj:`tgp.reduce` to obtain PyG
  Aggregation instances by string (e.g. :obj:`"sum"`, :obj:`"mean"`, :obj:`"lstm"`,
  :obj:`"set2set"`). Use with :obj:`AggrReduce` and :obj:`readout`. Parametrized
  aggregators accept kwargs such as :obj:`in_channels`, :obj:`out_channels`,
  :obj:`processing_steps`.
- **Readout** is **no longer a method** on pooler classes. Use
  :obj:`tgp.reduce.readout(x, reduce_op=..., batch=..., mask=...)` directly. When
  :obj:`reduce_op` is a string or a PyG Aggregation, readout uses :obj:`AggrReduce`
  internally with :obj:`so=None` (one cluster per graph). **Mask is only supported
  for dense x** (3D); pass :obj:`mask=None` for sparse (2D) x.
- **AggrReduce** supports :obj:`so=None` for graph-level readout (one cluster per
  graph, or single graph → one vector). Dense assignment is supported only via
  argmax (hard assignment) with a warning; for soft assignments use BaseReduce.
  **Forward** accepts optional :obj:`mask` (original nodes) only for batched (3D) x;
  resolution is mask arg → :obj:`so.in_mask` → all valid. Invalid mask (2D x or
  wrong shape) triggers a warning and is ignored (all nodes valid).
- **return_batched** is documented as applying only to the **unbatched (sparse) path**
  in all reduce operators; dense path always returns dense (same ndim as input).
- **Bug fixes:** AggrReduce dense path now returns 3D when input is 3D (fixes shape
  errors with batched dense poolers). Reshape after aggregation uses
  :obj:`x_pool.size(-1)` so Set2Set and other aggrs that change feature dim work
  correctly. LSTM (and similar) aggregation now receives index sorted by destination
  in the sparse path.

## Dense Pooling Modes (Intended Usage)

Dense poolers implement **two internal processing modes**:

1. **Batched mode (`batched=True`)**
   - Converts sparse inputs into dense padded tensors:
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

### Output format

`sparse_output` controls the **format** of the pooled adjacency:

- `sparse_output=True`: **block‑diagonal sparse** output (`edge_index`, `edge_weight`, `batch`).
- `sparse_output=False`: **batched dense** adjacency of shape `[B, K, K]`.

This flag determines the appropriate downstream MP/global pooling layers.

## Mask Handling (Dense Inputs)

- **External masks are supported only when inputs are already dense/padded**.
- When inputs are sparse, masks are produced internally during preprocessing.
- Dense poolers that accept `mask` document that it is only honored for pre‑padded inputs.

### SelectOutput: in_mask and out_mask

- **`SelectOutput.in_mask`** is an optional stored attribute (default `None`), set at
  init (explicit parameter or legacy `mask` in extra_args). **Batched only:** shape
  must be `[B, N]` (2D); 1D masks are not supported. Mask on **original nodes**. Used
  by `is_expressive` and by reduce when `so.s.dim() == 3` (batched dense).
- **`SelectOutput.out_mask`** is a **property** (not stored). It returns
  `(so.s.sum(dim=-2) > 0)` when `so.s.dim() == 3`, else `None`. Mask on **pooled
  nodes** `[B, K]`. For batched dense assignment only.

### PoolingOutput mask (variable supernode counts)

- **`PoolingOutput.mask`** is a **property** derived from `so.out_mask`: it returns
  `self.so.out_mask` when `so` is set, else `None`. No stored mask field; do not
  pass `mask=` when constructing `PoolingOutput`. Shape `[B, K]` when batched dense.
- Downstream layers (e.g. `DenseGCNConv`, global pool) use `out.mask` to ignore
  padded supernodes. For unbatched dense output with 2D `s`, `so.out_mask` is `None`;
  use `get_mask_from_dense_s(so.s, batch)` if a `[B, K]` mask is needed there.

### Readout (graph-level aggregation)

- **`readout(...)`** is the single graph-level readout function. Import from
  `tgp.reduce`: `from tgp.reduce import readout`. It infers sparse vs dense from
  the tensor shape: 2D `[N, F]` is treated as sparse (use `batch` for grouping);
  3D `[B, N, F]` as dense. **Node dimension is fixed:** `x` must be `[N, F]` or
  `[B, N, F]` (nodes on the second-to-last dimension); there is no `node_dim` parameter.
- **Reduce op:** Only PyG-style aggregators are supported (string alias or
  `torch_geometric.nn.aggr.Aggregation` instance). The `"any"` reduce is not
  supported; use `"sum"`, `"mean"`, `"max"`, `"min"`, or parametrized aggrs
  (e.g. `"lstm"`, `"set2set"`) via `get_aggr(reduce_op, **kwargs)`.
- **Mask:** All mask handling lives in `AggrReduce`. Readout forwards `mask` to
  the reducer; for sparse (2D) x or invalid mask shape, `AggrReduce` emits a
  warning and ignores the mask (all nodes valid). For dense x, `mask` has shape
  `[B, N]`.
- **Poolers no longer have a `readout` method.** Call `readout(x, reduce_op=...,
  batch=..., mask=...)` directly on the pooled node features (use `mask` only when
  `x` is 3D).
- **AggrReduce (mask application):** When a valid mask is provided for batched
  dense input, `AggrReduce` **selects only valid nodes** (sparse-like
  `x_valid`, `index_valid` from argmax over `so.s`) and runs the aggregator on
  them; masked-out nodes are not included (no zero-padding). For readout
  (`so=None`), valid nodes are extracted and assigned one cluster per graph.
- **AggrReduce (mask resolution):** `forward(..., mask=...)` uses (1) the `mask`
  argument if provided and valid, (2) else `so.in_mask` if `so` is not None and
  `so.s.dim() == 3`, (3) else all nodes valid. Mask is on **original nodes**.

## API / Behavior Changes

- **`sparse_output` replaces legacy output flags** across the codebase
  (e.g., `block_diags_output`, `unbatched_output`).
- **`DenseConnect`** now handles batched dense inputs and unbatched sparse inputs
  under one class (no separate unbatched class or helper file).
- **`DenseSelect` renamed to `MLPSelect`**, with module rename from
  `dense_select.py` → `mlp_select.py`.
- **`MLPSelect`/`DPSelect` now accept `batched_representation`** to emit either
  batched `[B, N, K]` or unbatched `[N, K]` assignments, and include `batch` in
  `SelectOutput` when operating unbatched.
- **`SparseBNPool` removed**; use `BNPool(batched=False)` or `get_pooler("bnpool_u")`.
- **`dense_conn_spt.py` removed**, all logic folded into `dense_conn.py`.
- **`get_pooler` now recognizes `_u` suffix** for dense poolers that implement the
  unbatched mode; unsupported `_u` names raise an error.

## Refactors & Shared Utilities

- **Post‑processing logic consolidated** in `tgp/utils/ops.py`:
  - `postprocess_adj_pool_dense(...)`
  - `postprocess_adj_pool_sparse(...)`
  These functions centralize self‑loop removal, degree normalization, and
  edge‑weight normalization.

- **Dense adjacency helpers** live in `tgp/utils/ops.py`:
  - `dense_to_block_diag`
  - `is_dense_adj`

## Pooler‑Specific Notes

- **BNPool**: unified dense/sparse behavior; batched/unbatched branches share a
  consistent interface. Unbatched path computes sparse loss.
- **BNPool (unbatched)** now supports a **negative‑sampling cap** via
  `num_neg_samples` to control memory/time when graphs are dense or `K` is large.
- **LaPooling**: default is `batched=True`; unbatched mode retained for memory efficiency.
  Unbatched selection now computes per‑graph similarity to reduce peak memory.
- **MinCutPooling**: unbatched mode (`batched=False`) computes sparse losses and can
  return either dense `[B, K, K]` or block‑diagonal sparse outputs based on
  `sparse_output`.
- **DiffPool**: unbatched mode (`batched=False` or `get_pooler("diff_u")`) uses sparse
  connectivity and sparse losses; output format is controlled by `sparse_output`.
- **JustBalancePooling (JBPool)**: unbatched mode (`batched=False` or `get_pooler("jb_u")`)
  uses sparse connectivity and sparse balance loss; output format is controlled by
  `sparse_output`.
- **DMoNPooling**: unbatched mode (`batched=False` or `get_pooler("dmon_u")`) uses sparse
  connectivity and sparse spectral/cluster/ortho losses; output format is controlled by
  `sparse_output`.
- **HOSCPooling**: unbatched mode (`batched=False` or `get_pooler("hosc_u")`) uses sparse
  connectivity and sparse HOSC/ortho losses; output format is controlled by
  `sparse_output`.
- **AsymCheegerCutPooling (ACC)**: unbatched mode (`batched=False` or `get_pooler("acc_u")`)
  uses sparse connectivity and sparse totvar/balance losses; output format is controlled by
  `sparse_output`.
- **NMFPooling**: both batched and unbatched modes are now supported.
  Unbatched mode operates on sparse connectivity without padding and can return
  dense or sparse pooled outputs. Pre‑coarsening returns sparse output by default
  for efficient downstream use.
- **EigenPooling**: added as a new pooling method with dedicated select/reduce/connect/lift
  operators for spectral-clustering-based hierarchical pooling.
- **SEP (Structural Entropy Guided Pooling)**: non-trainable, topology-driven clustering via
  coding tree; single-level forward yields depth-1 partition. Available via
  `get_pooler("sep")` and SRC pipeline; supports multilevel hierarchy in precoarsening.

## Bug Fixes

- **Spectral loss** (batched and unbatched) no longer returns NaN for graphs with zero edges;
  empty graphs now contribute zero loss and the result remains finite.
- **Weighted BCE reconstruction loss** now counts edges using a boolean mask to
  avoid mismatches when adjacency is non‑binary or has clamped edges.
- **Batched random assignment** in `get_random_map_mask` now handles graphs with
  zero kept nodes by falling back to global assignments.
- **AsymNorm and HOSC orthogonality losses** now handle single-node / single-cluster
  edge cases without NaNs or index errors, returning finite, well-defined values.
- **JustBalance (JB) loss** in batched mode now correctly applies per-graph
  normalization when a node mask is provided (e.g. variable-sized graphs with
  zero-padding).
- **AsymNorm (ACC) loss** in batched mode now computes correctly per graph when a
  node mask is provided: the batched path builds a flat assignment and batch
  vector from the mask and delegates to the unbatched implementation.

## Migration Notes

- **Readout:** Replace `global_reduce(x, ...)` and `dense_global_reduce(x, ...)`
  with `readout(x, ...)`. Replace `pooler.global_pool(x, ...)` with
  `readout(x, ...)`. Remove `node_dim` from pooler constructors and from
  KMIS if you used it.
- If you previously relied on `block_diags_output` or `unbatched_output`,
  update to `sparse_output`.
- Use `_u` pooler names (e.g., `bnpool_u`, `lap_u`) to select unbatched dense modes.
- Dense poolers should no longer require explicit calls to `preprocessing(...)`
  in user code; they perform it internally when `batched=True`.
- The PreCoarsening transform now takes a single argument, `poolers` (one
  pooler/config or a sequence for multilevel); update any code that passed
  pooler configuration under a different parameter name. The recursive
  depth argument is no longer supported: the number of levels is given by
  the length of `poolers`, so for three levels use e.g.
  `poolers=["ndp", "ndp", "ndp"]` instead of a
  depth parameter.
