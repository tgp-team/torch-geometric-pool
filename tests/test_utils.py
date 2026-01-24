"""Common test utilities for graph generation and testing.

This module provides reusable functions for creating test graphs of various types
in both sparse and dense formats. All functions are designed to be used across
multiple test files to reduce code duplication.

Graph Types:
- Chain graphs: Linear chain of connected nodes
- Erdos-Renyi graphs: Random graphs with edge probability p
- Barabasi-Albert graphs: Preferential attachment random graphs
- Grid graphs: 2D grid graphs with 4-connected neighbors

Each graph type has both sparse and dense versions.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import (
    barabasi_albert_graph,
    erdos_renyi_graph,
)


def set_test_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility in tests.

    Args:
        seed: Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_random_seed(seed: int = 42) -> None:
    """Convenience alias for set_test_seed(42). Use in fixtures for reproducible tests."""
    set_test_seed(seed)


# ============================================================================
# Batch data utilities
# ============================================================================


def make_small_batch_data(
    batch_size: int = 2,
    n_nodes: int = 4,
    n_features: int = 3,
    seed: Optional[int] = 42,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Create small batched dense data (x, adj, mask) for testing.

    Returns:
        Tuple of (x, adj, mask):
        - x: Node features [batch_size, n_nodes, n_features]
        - adj: Dense adjacency [batch_size, n_nodes, n_nodes]
        - mask: All ones [batch_size, n_nodes]
    """
    x, adj = make_dense_batched_graph(batch_size, n_nodes, n_features, seed=seed)
    mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    return x, adj, mask


def make_variable_size_batch_data(
    batch_size: int = 3,
    max_nodes: int = 5,
    n_features: int = 2,
    seed: Optional[int] = 42,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Create batched dense data with variable graph sizes via masks.

    Graph 0 has 3 nodes, graph 1 has 4, graph 2 has 5 (when batch_size=3, max_nodes=5).

    Returns:
        Tuple of (x, adj, mask):
        - x: [batch_size, max_nodes, n_features]
        - adj: [batch_size, max_nodes, max_nodes]
        - mask: [batch_size, max_nodes], True for valid nodes per graph
    """
    if seed is not None:
        set_test_seed(seed)
    x = torch.randn(batch_size, max_nodes, n_features)
    adj = torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float()
    adj = (((adj + adj.transpose(-1, -2)) / 2) > 0).float()

    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    for i in range(batch_size):
        n = min(3 + i, max_nodes)  # 3, 4, 5 for default
        mask[i, :n] = True
    return x, adj, mask


def make_single_sparse_graph_data(
    n_nodes: int,
    n_features: int,
    m: int = 2,
    seed: Optional[int] = 42,
) -> Data:
    """Create a single sparse graph as PyG Data (Barabasi-Albert)."""
    x, edge_index, _, _ = make_barabasi_albert_sparse(
        n_nodes, n_features, m=m, seed=seed
    )
    return Data(x=x, edge_index=edge_index)


# ============================================================================
# Chain Graphs
# ============================================================================


def make_chain_graph_sparse(
    N: int, F_dim: int, add_self_loops: bool = False, seed: Optional[int] = 42
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a sparse chain graph (linear path).

    Creates an undirected chain graph: 0-1-2-...-N-1

    Args:
        N: Number of nodes
        F_dim: Feature dimension
        add_self_loops: Whether to add self-loops to the graph
        seed: Random seed for feature generation

    Returns:
        Tuple of (x, edge_index, edge_weight, batch):
        - x: Node features [N, F_dim]
        - edge_index: Edge indices [2, E]
        - edge_weight: Edge weights [E]
        - batch: Batch vector [N] (all zeros for single graph)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create chain edges: (0,1), (1,2), ..., (N-2, N-1)
    row = torch.arange(N - 1, dtype=torch.long)
    col = row + 1
    # Make undirected
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)

    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)

    if add_self_loops:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_attr=edge_weight, num_nodes=N
        )

    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


def make_chain_graph_dense(
    N: int, F_dim: int, seed: Optional[int] = 42
) -> Tuple[Tensor, Tensor]:
    """Create a dense chain graph (linear path).

    Creates an undirected chain graph: 0-1-2-...-N-1

    Args:
        N: Number of nodes
        F_dim: Feature dimension
        seed: Random seed for feature generation

    Returns:
        Tuple of (x, adj):
        - x: Node features [N, F_dim]
        - adj: Dense adjacency matrix [N, N]
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn((N, F_dim), dtype=torch.float)

    # Create dense adjacency for chain
    adj = torch.zeros((N, N), dtype=torch.float)
    for i in range(N - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0  # Undirected

    return x, adj


def make_chain_edge_index(
    N: int = 3, add_self_loops: bool = False, seed: Optional[int] = 42
) -> Tensor:
    """Create edge_index for an N-node undirected chain. Optionally add self-loops."""
    _, edge_index, _, _ = make_chain_graph_sparse(
        N, F_dim=1, add_self_loops=add_self_loops, seed=seed
    )
    return edge_index


def make_simple_edge_index() -> Tensor:
    """Minimal 3-node undirected chain: 0-1-2."""
    return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)


def make_simple_undirected_graph(seed: Optional[int] = 42) -> Tensor:
    """4-node chain with self-loops. Returns edge_index."""
    _, edge_index, _, _ = make_chain_graph_sparse(
        N=4, F_dim=2, add_self_loops=True, seed=seed
    )
    return edge_index


def make_simple_undirected_edge_index(seed: Optional[int] = 42) -> Tensor:
    """5-node chain with self-loops. Returns edge_index."""
    _, edge_index, _, _ = make_chain_graph_sparse(
        N=5, F_dim=3, add_self_loops=True, seed=seed
    )
    return edge_index


# ============================================================================
# Erdos-Renyi Graphs
# ============================================================================


def make_erdos_renyi_sparse(
    N: int,
    F_dim: int,
    p: float = 0.5,
    num_disconnected: int = 0,
    seed: Optional[int] = 42,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a sparse Erdos-Renyi random graph.

    Args:
        N: Number of nodes
        F_dim: Feature dimension
        p: Edge probability (default: 0.5)
        num_disconnected: Number of isolated nodes to add (default: 0)
        seed: Random seed for graph and feature generation

    Returns:
        Tuple of (x, edge_index, edge_weight, batch):
        - x: Node features [N, F_dim]
        - edge_index: Edge indices [2, E]
        - edge_weight: Edge weights [E]
        - batch: Batch vector [N] (all zeros for single graph)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate ER graph
    edge_index = erdos_renyi_graph(N - num_disconnected, p, directed=False)

    # Add disconnected nodes by offsetting existing edges if needed
    # and ensuring disconnected nodes have no edges
    if num_disconnected > 0:
        # The ER graph is already generated for N - num_disconnected nodes
        # The disconnected nodes (N - num_disconnected, ..., N - 1) have no edges
        pass

    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


def make_erdos_renyi_dense(
    N: int,
    F_dim: int,
    p: float = 0.5,
    num_disconnected: int = 0,
    seed: Optional[int] = 42,
) -> Tuple[Tensor, Tensor]:
    """Create a dense Erdos-Renyi random graph.

    Args:
        N: Number of nodes
        F_dim: Feature dimension
        p: Edge probability (default: 0.5)
        num_disconnected: Number of isolated nodes to add (default: 0)
        seed: Random seed for graph and feature generation

    Returns:
        Tuple of (x, adj):
        - x: Node features [N, F_dim]
        - adj: Dense adjacency matrix [N, N]
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn((N, F_dim), dtype=torch.float)

    # Generate ER graph as sparse first, then convert to dense
    edge_index = erdos_renyi_graph(N - num_disconnected, p, directed=False)

    # Create dense adjacency
    adj = torch.zeros((N, N), dtype=torch.float)
    if edge_index.size(1) > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0  # Make symmetric

    # Ensure disconnected nodes have no edges (already zeros)
    # The last num_disconnected nodes will have no connections

    return x, adj


# ============================================================================
# Barabasi-Albert Graphs
# ============================================================================


def make_barabasi_albert_sparse(
    N: int, F_dim: int, m: int = 2, seed: Optional[int] = 42
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a sparse Barabasi-Albert random graph.

    Args:
        N: Number of nodes
        F_dim: Feature dimension
        m: Number of edges to attach from a new node to existing nodes (default: 2)
        seed: Random seed for graph and feature generation

    Returns:
        Tuple of (x, edge_index, edge_weight, batch):
        - x: Node features [N, F_dim]
        - edge_index: Edge indices [2, E]
        - edge_weight: Edge weights [E]
        - batch: Batch vector [N] (all zeros for single graph)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate BA graph
    edge_index = barabasi_albert_graph(N, m, directed=False)

    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


def make_barabasi_albert_dense(
    N: int, F_dim: int, m: int = 2, seed: Optional[int] = 42
) -> Tuple[Tensor, Tensor]:
    """Create a dense Barabasi-Albert random graph.

    Args:
        N: Number of nodes
        F_dim: Feature dimension
        m: Number of edges to attach from a new node to existing nodes (default: 2)
        seed: Random seed for graph and feature generation

    Returns:
        Tuple of (x, adj):
        - x: Node features [N, F_dim]
        - adj: Dense adjacency matrix [N, N]
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn((N, F_dim), dtype=torch.float)

    # Generate BA graph as sparse first, then convert to dense
    edge_index = barabasi_albert_graph(N, m, directed=False)

    # Create dense adjacency
    adj = torch.zeros((N, N), dtype=torch.float)
    if edge_index.size(1) > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0  # Make symmetric

    return x, adj


# ============================================================================
# Grid Graphs
# ============================================================================


def make_grid_graph_sparse(
    rows: int, cols: int, F_dim: int, seed: Optional[int] = 42
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a sparse 2D grid graph.

    Creates a grid with 4-connected neighbors (up, down, left, right).

    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        F_dim: Feature dimension
        seed: Random seed for feature generation

    Returns:
        Tuple of (x, edge_index, edge_weight, batch):
        - x: Node features [rows*cols, F_dim]
        - edge_index: Edge indices [2, E]
        - edge_weight: Edge weights [E]
        - batch: Batch vector [rows*cols] (all zeros for single graph)
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = rows * cols
    edge_list = []

    # Horizontal edges
    for i in range(rows):
        for j in range(cols - 1):
            node = i * cols + j
            edge_list.append([node, node + 1])
            edge_list.append([node + 1, node])  # Undirected

    # Vertical edges
    for i in range(rows - 1):
        for j in range(cols):
            node = i * cols + j
            edge_list.append([node, node + cols])
            edge_list.append([node + cols, node])  # Undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    E = edge_index.size(1)

    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


def make_grid_graph_dense(
    rows: int, cols: int, F_dim: int, seed: Optional[int] = 42
) -> Tuple[Tensor, Tensor]:
    """Create a dense 2D grid graph.

    Creates a grid with 4-connected neighbors (up, down, left, right).

    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        F_dim: Feature dimension
        seed: Random seed for feature generation

    Returns:
        Tuple of (x, adj):
        - x: Node features [rows*cols, F_dim]
        - adj: Dense adjacency matrix [rows*cols, rows*cols]
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = rows * cols
    x = torch.randn((N, F_dim), dtype=torch.float)
    adj = torch.zeros((N, N), dtype=torch.float)

    # Horizontal edges
    for i in range(rows):
        for j in range(cols - 1):
            node = i * cols + j
            adj[node, node + 1] = 1.0
            adj[node + 1, node] = 1.0  # Undirected

    # Vertical edges
    for i in range(rows - 1):
        for j in range(cols):
            node = i * cols + j
            adj[node, node + cols] = 1.0
            adj[node + cols, node] = 1.0  # Undirected

    return x, adj


# ============================================================================
# Batched Dense Graphs
# ============================================================================


def make_dense_batched_graph(
    batch_size: int,
    n_nodes: int,
    n_features: int,
    seed: Optional[int] = 42,
) -> Tuple[Tensor, Tensor]:
    """Create a batch of dense graphs.

    Args:
        batch_size: Number of graphs in the batch
        n_nodes: Number of nodes per graph
        n_features: Feature dimension
        seed: Random seed for generation

    Returns:
        Tuple of (x, adj):
        - x: Node features [batch_size, n_nodes, n_features]
        - adj: Dense adjacency matrices [batch_size, n_nodes, n_nodes]
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn(batch_size, n_nodes, n_features)
    adj = torch.randint(0, 2, (batch_size, n_nodes, n_nodes))

    # Ensure at least one edge for each node
    isolated = (adj.sum(-1) == 0) & (adj.sum(-2) == 0)
    adj[isolated, 0] = 1  # Add self-loop or connection

    # Make symmetric
    adj = (((adj + adj.transpose(-1, -2)) / 2) > 0).float()

    return x, adj


# ============================================================================
# Mixed Graph Batch
# ============================================================================


def make_mixed_graph_batch_sparse(
    N: int, F_dim: int, seed: Optional[int] = 42
) -> Batch:
    """Create a batched graph containing one graph of each type.

    Creates a batch with:
    - One chain graph
    - One Erdos-Renyi graph
    - One Barabasi-Albert graph
    - One grid graph (approximately N nodes)

    Args:
        N: Approximate number of nodes per graph
        F_dim: Feature dimension
        seed: Random seed for generation

    Returns:
        Batch: PyTorch Geometric Batch object containing all graphs
    """
    if seed is not None:
        set_test_seed(seed)

    data_list = []

    # Chain graph
    x1, edge_index1, edge_weight1, batch1 = make_chain_graph_sparse(
        N, F_dim, add_self_loops=False, seed=None
    )
    data_list.append(Data(x=x1, edge_index=edge_index1, edge_attr=edge_weight1))

    # Erdos-Renyi graph
    x2, edge_index2, edge_weight2, batch2 = make_erdos_renyi_sparse(
        N, F_dim, p=0.3, num_disconnected=2, seed=None
    )
    data_list.append(Data(x=x2, edge_index=edge_index2, edge_attr=edge_weight2))

    # Barabasi-Albert graph
    x3, edge_index3, edge_weight3, batch3 = make_barabasi_albert_sparse(
        N, F_dim, m=2, seed=None
    )
    data_list.append(Data(x=x3, edge_index=edge_index3, edge_attr=edge_weight3))

    # Grid graph (find grid dimensions that give approximately N nodes)
    rows = int(np.sqrt(N))
    cols = (N + rows - 1) // rows  # Ceiling division
    x4, edge_index4, edge_weight4, batch4 = make_grid_graph_sparse(
        rows, cols, F_dim, seed=None
    )
    data_list.append(Data(x=x4, edge_index=edge_index4, edge_attr=edge_weight4))

    return Batch.from_data_list(data_list)


def make_mixed_graph_batch_dense(
    N: int, F_dim: int, seed: Optional[int] = 42
) -> Tuple[list, list, list]:
    """Create a list of dense graphs, one of each type.

    Creates lists containing:
    - One chain graph
    - One Erdos-Renyi graph
    - One Barabasi-Albert graph
    - One grid graph (approximately N nodes)

    Args:
        N: Approximate number of nodes per graph
        F_dim: Feature dimension
        seed: Random seed for generation

    Returns:
        Tuple of (x_list, adj_list, batch_list):
        - x_list: List of node feature tensors
        - adj_list: List of dense adjacency matrices
        - batch_list: List of batch vectors
    """
    if seed is not None:
        set_test_seed(seed)

    x_list = []
    adj_list = []
    batch_list = []

    # Chain graph
    x1, adj1 = make_chain_graph_dense(N, F_dim, seed=None)
    x_list.append(x1)
    adj_list.append(adj1)
    batch_list.append(torch.zeros(N, dtype=torch.long))

    # Erdos-Renyi graph
    x2, adj2 = make_erdos_renyi_dense(N, F_dim, p=0.3, num_disconnected=2, seed=None)
    x_list.append(x2)
    adj_list.append(adj2)
    batch_list.append(torch.zeros(N, dtype=torch.long))

    # Barabasi-Albert graph
    x3, adj3 = make_barabasi_albert_dense(N, F_dim, m=2, seed=None)
    x_list.append(x3)
    adj_list.append(adj3)
    batch_list.append(torch.zeros(N, dtype=torch.long))

    # Grid graph
    rows = int(np.sqrt(N))
    cols = (N + rows - 1) // rows
    x4, adj4 = make_grid_graph_dense(rows, cols, F_dim, seed=None)
    x_list.append(x4)
    adj_list.append(adj4)
    batch_list.append(torch.zeros(rows * cols, dtype=torch.long))

    return x_list, adj_list, batch_list


# ============================================================================
# Canonical pooler test graphs (same for all poolers)
# ============================================================================

POOLER_TEST_N = 6
POOLER_TEST_F = 3
POOLER_TEST_B = 2


def make_pooler_test_graph_sparse(
    seed: int = 42,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Canonical single sparse graph for all pooler tests.

    Returns (x, edge_index, edge_weight, batch). Chain, N=6, F=3, self-loops.
    """
    return make_chain_graph_sparse(
        N=POOLER_TEST_N,
        F_dim=POOLER_TEST_F,
        add_self_loops=True,
        seed=seed,
    )


def make_pooler_test_graph_dense(
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """Canonical single dense graph (B=1) for all pooler tests.

    Returns (x, adj). x [1, N, F], adj [1, N, N].
    """
    return make_dense_batched_graph(1, POOLER_TEST_N, POOLER_TEST_F, seed=seed)


def make_pooler_test_graph_sparse_batch(seed: int = 42) -> Batch:
    """Canonical batched sparse graph for all pooler tests.

    Returns PyG Batch (chain, ER, BA, grid).
    """
    return make_mixed_graph_batch_sparse(
        N=POOLER_TEST_N, F_dim=POOLER_TEST_F, seed=seed
    )


def make_pooler_test_graph_dense_batch(
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """Canonical batched dense graph for all pooler tests.

    Returns (x, adj). x [B, N, F], adj [B, N, N].
    """
    return make_dense_batched_graph(
        POOLER_TEST_B, POOLER_TEST_N, POOLER_TEST_F, seed=seed
    )


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_graph_output(
    out, expected_shape: Optional[Tuple] = None, expected_type: Optional[type] = None
) -> bool:
    """Validate graph output structure.

    Args:
        out: Output object to validate
        expected_shape: Expected shape tuple (optional)
        expected_type: Expected type (optional)

    Returns:
        True if validation passes, raises AssertionError otherwise
    """
    if expected_type is not None:
        assert isinstance(out, expected_type), (
            f"Expected type {expected_type}, got {type(out)}"
        )

    if expected_shape is not None:
        if hasattr(out, "shape"):
            assert out.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {out.shape}"
            )
        elif hasattr(out, "size"):
            assert out.size() == expected_shape, (
                f"Expected shape {expected_shape}, got {out.size()}"
            )

    return True
