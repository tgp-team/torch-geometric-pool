"""Shared helpers for dense loss equivalence tests (batched dense vs sparse/unbatched)."""

from typing import Optional, Tuple

import torch


def _dense_batched_to_sparse_unbatched(
    adj: torch.Tensor,
    S: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dense batched (adj [B,N,N], S [B,N,K]) to sparse unbatched form.

    If mask is None, all nodes are used (padded batch: N_total = B*N).
    If mask (B, N) is provided, only masked nodes are included (real nodes only);
    then N_total = mask.sum(), and edge_index only has edges between real nodes.

    Returns edge_index [2, E], edge_weight [E], S_flat [N_total, K], batch [N_total].
    """
    B, N, K = S.shape
    device = S.device

    if mask is None:
        edge_index_list = []
        edge_weight_list = []
        batch_list = []
        for b in range(B):
            nz = adj[b].nonzero(as_tuple=False)
            if nz.size(0) == 0:
                edge_index_list.append(
                    torch.zeros(2, 0, dtype=torch.long, device=device)
                )
                edge_weight_list.append(torch.zeros(0, device=device))
            else:
                row, col = nz[:, 0], nz[:, 1]
                edge_index_b = torch.stack([row, col], dim=0)
                edge_weight_b = adj[b][row, col]
                offset = b * N
                edge_index_b = edge_index_b + offset
                edge_index_list.append(edge_index_b)
                edge_weight_list.append(edge_weight_b)
            batch_list.append(torch.full((N,), b, dtype=torch.long, device=device))
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_weight = torch.cat(edge_weight_list)
        batch = torch.cat(batch_list)
        S_flat = S.reshape(B * N, K)
        return edge_index, edge_weight, S_flat, batch

    # Masked path: only real nodes, so sparse side matches "no padding" semantics
    S_list = []
    batch_list = []
    edge_index_list = []
    edge_weight_list = []
    offset = 0
    for b in range(B):
        real = mask[b].nonzero(as_tuple=True)[0]  # [n_b]
        n_b = real.size(0)
        S_list.append(S[b][real])
        batch_list.append(torch.full((n_b,), b, dtype=torch.long, device=device))
        adj_b = adj[b][real][:, real]
        nz = adj_b.nonzero(as_tuple=False)
        if nz.size(0) == 0:
            edge_index_list.append(torch.zeros(2, 0, dtype=torch.long, device=device))
            edge_weight_list.append(torch.zeros(0, device=device))
        else:
            row, col = nz[:, 0], nz[:, 1]
            edge_index_list.append(torch.stack([row + offset, col + offset], dim=0))
            edge_weight_list.append(adj_b[row, col])
        offset += n_b
    S_flat = torch.cat(S_list, dim=0)
    batch = torch.cat(batch_list)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_weight = torch.cat(edge_weight_list)
    return edge_index, edge_weight, S_flat, batch


def _make_dense_batch(
    B: int, N: int, K: int, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same-size batch: all graphs have N nodes. Returns adj (B, N, N), S (B, N, K)."""
    torch.manual_seed(seed)
    adj = torch.rand(B, N, N)
    adj = adj + adj.transpose(-2, -1)
    for b in range(B):
        adj[b].fill_diagonal_(0)
    S = torch.randn(B, N, K)
    S = torch.softmax(S, dim=-1)
    return adj, S


def _make_dense_batch_variable_sizes(
    K: int = 3,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch of 3 graphs with different sizes, padded to a common N_max.

    Graph 0: 3 nodes, graph 1: 5 nodes, graph 2: 4 nodes -> padded to N_max=5.
    Returns adj (3, 5, 5), S (3, 5, K), mask (3, 5) with True only for real nodes.
    """
    torch.manual_seed(seed)
    sizes = [3, 5, 4]
    B = len(sizes)
    N_max = max(sizes)

    adj = torch.zeros(B, N_max, N_max)
    S = torch.zeros(B, N_max, K)
    mask = torch.zeros(B, N_max, dtype=torch.bool)

    for b in range(B):
        n = sizes[b]
        # Build small adj and S for this graph
        a = torch.rand(n, n)
        a = a + a.t()
        a.fill_diagonal_(0)
        adj[b, :n, :n] = a
        s = torch.randn(n, K)
        S[b, :n, :] = torch.softmax(s, dim=-1)
        mask[b, :n] = True
        # Padding rows of S are left zero, consistent with MLPSelect (mask zeros out
        # padded positions), so batched losses see zero rows for invalid nodes.

    return adj, S, mask
