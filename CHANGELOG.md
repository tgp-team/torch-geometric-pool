# Torch Geometric Pool — Changelog (Refactor Edition)

This changelog summarizes the recent architectural refactor and behavioral updates.
It focuses on public‑facing API/behavior, intended usage, and design tradeoffs.

## Highlights

- **Unified dense pooling interface**: Dense poolers now accept raw sparse inputs and
  handle batching/densification internally when `batched=True`.
- **Explicit output format control**: A single flag, `sparse_output`, now determines
  whether pooled adjacency is returned as block‑diagonal sparse or batched dense.
- **BNPool unified**: Dense and sparse BNPool variants are merged into one class with
  batched/unbatched branches and consistent outputs.
- **DenseConnect consolidated**: All dense connect logic (batched and unbatched paths)
  lives in `DenseConnect`; redundant variants were removed.

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

## API / Behavior Changes

- **`sparse_output` replaces legacy output flags** across the codebase
  (e.g., `block_diags_output`, `unbatched_output`).
- **`DenseConnect`** now handles batched dense inputs and unbatched sparse inputs
  under one class (no separate unbatched class or helper file).
- **`DenseSelect` renamed to `MLPSelect`**, with module rename from
  `dense_select.py` → `mlp_select.py`.
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
- **LaPooling**: default is `batched=True`; unbatched mode retained for memory efficiency.
  Unbatched selection now computes per‑graph similarity to reduce peak memory.
- **NMFPooling**: batched mode supported; unbatched path still not implemented.
  Pre‑coarsening returns sparse output by default for efficient downstream use.

## Bug Fixes

- **Weighted BCE reconstruction loss** now counts edges using a boolean mask to
  avoid mismatches when adjacency is non‑binary or has clamped edges.
- **Batched random assignment** in `get_random_map_mask` now handles graphs with
  zero kept nodes by falling back to global assignments.

## Migration Notes

- If you previously relied on `block_diags_output` or `unbatched_output`,
  update to `sparse_output`.
- Use `_u` pooler names (e.g., `bnpool_u`, `lap_u`) to select unbatched dense modes.
- Dense poolers should no longer require explicit calls to `preprocessing(...)`
  in user code; they perform it internally when `batched=True`.
