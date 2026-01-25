#!/usr/bin/env python3
"""Systematic Time and Memory Benchmark for Pooling Operators

This script benchmarks all poolers to measure time and memory usage on batches
of random graphs with variable sizes.

Usage:
    python examples/time_and_mem_test.py
"""

import gc
import os
import random
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch_geometric import seed_everything
from torch_geometric.data import Batch, Data
from torch_geometric.utils import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    to_undirected,
)

try:
    import psutil
except ImportError:
    psutil = None

from tgp.poolers import get_pooler, pooler_map

# Suppress warnings for cleaner output
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Sparse CSR tensor support is in beta.*"
)

# ============================================================================
# Configuration (hardcoded like classification.py)
# ============================================================================

NUM_GRAPHS = 4  # Number of graphs in batch
MIN_SIZE = 50  # Minimum nodes per graph
MAX_SIZE = 2000  # Maximum nodes per graph
NUM_ITERATIONS = 10  # Number of iterations to average
F_DIM = 16  # Feature dimension
POOLERS_TO_TEST = [
    "bnpool",
    "bnpool_u",
    "mincut",
    "lap",
    "lap_u",
]  # None  # None = all poolers, or specify list like ['topk', 'sag', 'diff']

# Common pooler parameters
PARAMS = {
    "cached": False,
    "lift": "precomputed",
    "s_inv_op": "transpose",
    "lift_red_op": "mean",
    "loss_coeff": 1.0,
    "k": 10,  # Will be adjusted per graph size
    "order_k": 2,
    "ratio": 0.25,
    "remove_self_loops": True,
    "scorer": "degree",
    "reduce": "sum",
    "block_diags_output": False,
}

# ============================================================================
# Data Structures and Helper Functions
# ============================================================================


@dataclass
class MemoryStats:
    alloc_total: int
    alloc_delta: int
    resv_total: Optional[int] = None
    resv_delta: Optional[int] = None


@dataclass
class BenchmarkResult:
    forward_time: float
    backward_time: float
    forward_memory: MemoryStats
    backward_memory: MemoryStats


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.2f}µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def format_memory(bytes_val: float) -> str:
    """Format memory in human-readable format."""
    if bytes_val < 1024:
        return f"{bytes_val:.0f}B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f}KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.2f}MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f}GB"


def format_memory_signed(bytes_val: float) -> str:
    """Format signed memory values in human-readable format."""
    if bytes_val < 0:
        return f"-{format_memory(abs(bytes_val))}"
    return format_memory(bytes_val)


def format_memory_pair(alloc_bytes: float, resv_bytes: Optional[float]) -> str:
    """Format memory as allocated/reserved when reserved is available."""
    if resv_bytes is None:
        return format_memory_signed(alloc_bytes)
    return f"{format_memory_signed(alloc_bytes)}/{format_memory_signed(resv_bytes)}"


# ============================================================================
# Graph Generation (from tests/test_utils.py)
# ============================================================================


def set_test_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass  # numpy not required for basic functionality
    random.seed(seed)


def make_erdos_renyi_sparse(
    N: int,
    F_dim: int,
    p: float = 0.5,
    num_disconnected: int = 0,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a sparse Erdos-Renyi random graph."""
    if seed is not None:
        torch.manual_seed(seed)

    # Generate ER graph
    edge_index = erdos_renyi_graph(N - num_disconnected, p, directed=False)

    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


def make_barabasi_albert_sparse(
    N: int, F_dim: int, m: int = 2, seed: Optional[int] = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a sparse Barabasi-Albert random graph."""
    if seed is not None:
        torch.manual_seed(seed)

    # Generate BA graph
    edge_index = barabasi_albert_graph(N, m)
    edge_index = to_undirected(edge_index)

    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


def generate_batch_with_variable_sizes(
    num_graphs: int,
    min_size: int,
    max_size: int,
    F_dim: int,
    seed: Optional[int] = 42,
) -> Batch:
    """Create a batch of graphs with randomly sampled sizes between min_size and max_size.

    Args:
        num_graphs: Number of graphs in the batch
        min_size: Minimum number of nodes per graph
        max_size: Maximum number of nodes per graph
        F_dim: Feature dimension
        seed: Random seed for reproducibility

    Returns:
        Batch: PyTorch Geometric Batch object containing all graphs
    """
    if seed is not None:
        set_test_seed(seed)
        random.seed(seed)

    data_list = []

    for i in range(num_graphs):
        # Randomly sample graph size between min_size and max_size
        N = random.randint(min_size, max_size)

        # Alternate between Erdos-Renyi and Barabasi-Albert graphs
        if i % 2 == 0:
            x, edge_index, edge_weight, batch = make_erdos_renyi_sparse(
                N, F_dim, p=0.3, num_disconnected=0, seed=None
            )
        else:
            x, edge_index, edge_weight, batch = make_barabasi_albert_sparse(
                N, F_dim, m=2, seed=None
            )

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_weight))

    return Batch.from_data_list(data_list)


# ============================================================================
# Benchmarking
# ============================================================================


def benchmark_pooler(
    pooler_name: str,
    batch: Batch,
    device: torch.device,
    num_iterations: int,
    params: Dict,
) -> BenchmarkResult:
    """Benchmark a single pooler on a batch of graphs (forward + backward).

    Args:
        pooler_name: Name of the pooler
        batch: Batch of graphs to test
        device: Device to run on (CPU or CUDA)
        num_iterations: Number of iterations to average
        params: Pooler parameters

    Returns:
        BenchmarkResult with forward/backward times and memories
    """
    # Move batch to device
    batch = batch.to(device)
    x = batch.x
    edge_index = batch.edge_index
    edge_weight = getattr(batch, "edge_attr", None)
    if edge_weight is None:
        edge_weight = getattr(batch, "edge_weight", None)
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device, dtype=torch.float
        )

    # Get number of features and adjust params
    num_features = x.size(1)
    pooler_params = params.copy()
    pooler_params["in_channels"] = num_features

    # Adjust k based on average graph size
    num_nodes = x.size(0)
    avg_nodes_per_graph = (
        num_nodes / batch.num_graphs if batch.num_graphs > 0 else num_nodes
    )
    pooler_params["k"] = max(
        1, int(avg_nodes_per_graph * pooler_params.get("ratio", 0.25))
    )

    # Create pooler
    pooler = get_pooler(pooler_name, **pooler_params)
    pooler = pooler.to(device)
    pooler.train()  # Set to train mode for gradients

    process = None
    if device.type == "cpu" and psutil is not None:
        process = psutil.Process(os.getpid())

    # Warm-up runs (with gradients)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(2):
        pooler.zero_grad(set_to_none=True)
        x_warmup = x.clone().requires_grad_(True)
        out = pooler(
            x=x_warmup, adj=edge_index, edge_weight=edge_weight, batch=batch.batch
        )
        loss = out.x.sum()
        if hasattr(out, "loss") and out.loss is not None:
            loss = loss + sum(out.get_loss_value())
        loss.backward()
        x_warmup.grad = None
        del out, loss, x_warmup
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Clean up warm-up memory
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "cpu" and psutil is None:
        tracemalloc.start()

    # Benchmark forward and backward passes
    forward_times = []
    backward_times = []
    forward_peak_allocated = 0
    forward_peak_reserved = 0
    forward_peak_alloc_delta = 0
    forward_peak_resv_delta = 0
    forward_rss_peak = 0
    forward_rss_delta = 0
    backward_peak_allocated = 0
    backward_peak_reserved = 0
    backward_peak_alloc_delta = 0
    backward_peak_resv_delta = 0
    backward_rss_peak = 0
    backward_rss_delta = 0

    for _ in range(num_iterations):
        pooler.zero_grad(set_to_none=True)
        x_input = x.clone().requires_grad_(True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            fwd_base_alloc = torch.cuda.memory_allocated(device)
            fwd_base_resv = torch.cuda.memory_reserved(device)
        elif process is not None:
            gc.collect()
            fwd_base_rss = process.memory_info().rss
        else:
            tracemalloc.reset_peak()
            fwd_base_current, _ = tracemalloc.get_traced_memory()

        start = time.perf_counter()
        out = pooler(
            x=x_input, adj=edge_index, edge_weight=edge_weight, batch=batch.batch
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)

        # Track memory after forward
        if device.type == "cuda":
            fwd_peak_alloc = torch.cuda.max_memory_allocated(device)
            fwd_peak_resv = torch.cuda.max_memory_reserved(device)
            forward_peak_allocated = max(forward_peak_allocated, fwd_peak_alloc)
            forward_peak_reserved = max(forward_peak_reserved, fwd_peak_resv)
            forward_peak_alloc_delta = max(
                forward_peak_alloc_delta, fwd_peak_alloc - fwd_base_alloc
            )
            forward_peak_resv_delta = max(
                forward_peak_resv_delta, fwd_peak_resv - fwd_base_resv
            )
        elif process is not None:
            fwd_rss = process.memory_info().rss
            forward_rss_peak = max(forward_rss_peak, fwd_rss)
            forward_rss_delta = max(forward_rss_delta, fwd_rss - fwd_base_rss)
        else:
            _, fwd_peak = tracemalloc.get_traced_memory()
            forward_rss_peak = max(forward_rss_peak, fwd_peak)
            forward_rss_delta = max(forward_rss_delta, fwd_peak - fwd_base_current)

        # Compute loss for backward
        loss = out.x.sum()
        if hasattr(out, "loss") and out.loss is not None:
            loss = loss + sum(out.get_loss_value())

        # Benchmark backward pass
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            bwd_base_alloc = torch.cuda.memory_allocated(device)
            bwd_base_resv = torch.cuda.memory_reserved(device)
        elif process is not None:
            gc.collect()
            bwd_base_rss = process.memory_info().rss
        else:
            tracemalloc.reset_peak()
            bwd_base_current, _ = tracemalloc.get_traced_memory()

        start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)

        # Track memory after backward
        if device.type == "cuda":
            bwd_peak_alloc = torch.cuda.max_memory_allocated(device)
            bwd_peak_resv = torch.cuda.max_memory_reserved(device)
            backward_peak_allocated = max(backward_peak_allocated, bwd_peak_alloc)
            backward_peak_reserved = max(backward_peak_reserved, bwd_peak_resv)
            backward_peak_alloc_delta = max(
                backward_peak_alloc_delta, bwd_peak_alloc - bwd_base_alloc
            )
            backward_peak_resv_delta = max(
                backward_peak_resv_delta, bwd_peak_resv - bwd_base_resv
            )
        elif process is not None:
            bwd_rss = process.memory_info().rss
            backward_rss_peak = max(backward_rss_peak, bwd_rss)
            backward_rss_delta = max(backward_rss_delta, bwd_rss - bwd_base_rss)
        else:
            _, bwd_peak = tracemalloc.get_traced_memory()
            backward_rss_peak = max(backward_rss_peak, bwd_peak)
            backward_rss_delta = max(backward_rss_delta, bwd_peak - bwd_base_current)

        # Cleanup for next iteration
        x_input.grad = None
        del out, loss, x_input

    # Get peak memory for forward/backward
    if device.type == "cuda":
        forward_memory = MemoryStats(
            alloc_total=forward_peak_allocated,
            alloc_delta=max(0, forward_peak_alloc_delta),
            resv_total=forward_peak_reserved,
            resv_delta=max(0, forward_peak_resv_delta),
        )
        backward_memory = MemoryStats(
            alloc_total=backward_peak_allocated,
            alloc_delta=max(0, backward_peak_alloc_delta),
            resv_total=backward_peak_reserved,
            resv_delta=max(0, backward_peak_resv_delta),
        )
    else:
        forward_memory = MemoryStats(
            alloc_total=max(0, forward_rss_peak),
            alloc_delta=max(0, forward_rss_delta),
        )
        backward_memory = MemoryStats(
            alloc_total=max(0, backward_rss_peak),
            alloc_delta=max(0, backward_rss_delta),
        )

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    # Cleanup
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu" and psutil is None:
        tracemalloc.stop()

    return BenchmarkResult(
        forward_time=avg_forward_time,
        backward_time=avg_backward_time,
        forward_memory=forward_memory,
        backward_memory=backward_memory,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    """Main benchmarking function."""
    # Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of graphs per batch: {NUM_GRAPHS}")
    print(f"Graph size range: {MIN_SIZE} - {MAX_SIZE} nodes")
    print(f"Iterations per test: {NUM_ITERATIONS}")
    print()

    # Get poolers to test
    if POOLERS_TO_TEST is None:
        poolers_to_test = list(pooler_map.keys())
        # Skip problematic poolers
        if "pan" in poolers_to_test:
            poolers_to_test.remove("pan")
            print("Skipping PAN (requires special setup)")
    else:
        poolers_to_test = POOLERS_TO_TEST

    print(f"Testing {len(poolers_to_test)} poolers: {', '.join(poolers_to_test)}\n")
    print("=" * 100)

    # Generate batch once
    print("Generating batch of random graphs...")
    batch = generate_batch_with_variable_sizes(
        NUM_GRAPHS, MIN_SIZE, MAX_SIZE, F_DIM, seed=42
    )
    print(f"Batch created: {batch.num_graphs} graphs, {batch.num_nodes} total nodes")
    print(f"Graph sizes: {[batch[i].num_nodes for i in range(batch.num_graphs)]}")
    print()

    # Results storage
    results = []

    # Benchmark each pooler
    for i, pooler_name in enumerate(poolers_to_test, 1):
        print(f"\n[{i}/{len(poolers_to_test)}] Benchmarking: {pooler_name.upper()}")
        print("-" * 100)

        try:
            result = benchmark_pooler(
                pooler_name, batch, device, NUM_ITERATIONS, PARAMS
            )

            results.append(
                {
                    "pooler": pooler_name,
                    "device": str(device),
                    "result": result,
                }
            )

            print(f"  ✓ Forward Time: {format_time(result.forward_time)}")
            print(f"  ✓ Backward Time: {format_time(result.backward_time)}")
            print(
                f"  ✓ Forward Memory A/R: {format_memory_pair(result.forward_memory.alloc_total, result.forward_memory.resv_total)}"
            )
            print(
                f"  ✓ Forward Memory Delta A/R: {format_memory_pair(result.forward_memory.alloc_delta, result.forward_memory.resv_delta)}"
            )
            print(
                f"  ✓ Backward Memory A/R: {format_memory_pair(result.backward_memory.alloc_total, result.backward_memory.resv_total)}"
            )
            print(
                f"  ✓ Backward Memory Delta A/R: {format_memory_pair(result.backward_memory.alloc_delta, result.backward_memory.resv_delta)}"
            )

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            raise  # Stop on error as requested

    # Summary report
    print("\n" + "=" * 120)
    print("SUMMARY REPORT")
    print("=" * 120)
    print(
        f"\n{'Pooler':<15} {'Device':<10} {'Fwd Time':<12} {'Bwd Time':<12} {'Fwd Mem (A/R)':<20} {'Bwd Mem (A/R)':<20}"
    )
    print("-" * 120)

    for result in results:
        r = result["result"]
        print(
            f"{result['pooler']:<15} "
            f"{result['device']:<10} "
            f"{format_time(r.forward_time):<12} "
            f"{format_time(r.backward_time):<12} "
            f"{format_memory_pair(r.forward_memory.alloc_total, r.forward_memory.resv_total):<20} "
            f"{format_memory_pair(r.backward_memory.alloc_total, r.backward_memory.resv_total):<20}"
        )

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
