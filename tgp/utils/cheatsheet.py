"""Cheatsheet utilities for checking pooler properties."""

import re

import torch
from torch_sparse import SparseTensor

from tgp.poolers import get_pooler, pooler_map

std_typing = __import__("typing", fromlist=["Dict", "List", "Optional", "Tuple"])
Dict = std_typing.Dict
List = std_typing.List
Optional = std_typing.Optional
Tuple = std_typing.Tuple

# Constants for testing poolers - these ensure consistent instantiation
POOLER_TEST_PARAMS = {"in_channels": 16, "ratio": 0.5, "k": 8, "scorer": "degree"}

# Test data for pooler instantiation
TEST_X = torch.randn(10, 16)  # 10 nodes, 16 features
TEST_EDGE_INDEX = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long
)
TEST_EDGE_WEIGHT = torch.ones(TEST_EDGE_INDEX.size(1))
TEST_ADJ = SparseTensor.from_edge_index(TEST_EDGE_INDEX, TEST_EDGE_WEIGHT)


def supports_sparse(pooler_name: str) -> bool:
    """Check if a pooler supports sparse operations.

    A pooler is considered sparse if it returns sparse connectivity matrices
    (not dense adjacency matrices).

    Args:
        pooler_name: Name of the pooler to check.

    Returns:
        True if the pooler is sparse, False if it's dense.
    """
    try:
        pooler = get_pooler(pooler_name, **POOLER_TEST_PARAMS)
        # Check if pooler operates in dense batched mode
        return not getattr(pooler, "is_dense_batched", False)
    except Exception:
        return False


def supports_trainable(pooler_name: str) -> bool:
    """Check if a pooler has trainable parameters.

    Args:
        pooler_name: Name of the pooler to check.

    Returns:
        True if the pooler has trainable parameters, False otherwise.
    """
    try:
        pooler = get_pooler(pooler_name, **POOLER_TEST_PARAMS)
        return pooler.is_trainable
    except Exception:
        return False


def supports_aux_loss(pooler_name: str) -> bool:
    """Check if a pooler supports auxiliary loss computation.

    Args:
        pooler_name: Name of the pooler to check.

    Returns:
        True if the pooler computes auxiliary losses, False otherwise.
    """
    try:
        pooler = get_pooler(pooler_name, **POOLER_TEST_PARAMS)
        return pooler.has_loss
    except Exception:
        return False


def extract_paper_links(pooler_class) -> List[str]:
    """Extract all unique paper links from a pooler class docstring.

    Args:
        pooler_class: The pooler class to extract the paper links from.

    Returns:
        List of unique paper links found in the docstring, preserving order of first occurrence.
    """
    if not pooler_class.__doc__:
        return []

    links = []
    seen_links = set()

    # Pattern to match markdown-style links or reStructuredText links
    rst_pattern = r"`[^`]*?\s*<(https?://[^>]+)>`_"
    rst_matches = re.findall(rst_pattern, pooler_class.__doc__)

    # Add reStructuredText links first (they have higher priority)
    for link in rst_matches:
        if link not in seen_links:
            links.append(link)
            seen_links.add(link)

    # Fallback: look for raw URLs that aren't already captured
    url_pattern = r'https?://[^\s<>"`]+'
    url_matches = re.findall(url_pattern, pooler_class.__doc__)

    # Only add URLs that aren't already in our links list
    for url in url_matches:
        if url not in seen_links:
            links.append(url)
            seen_links.add(url)

    return links


def get_pooler_cheatsheet() -> List[Tuple[str, str, bool, bool, bool, List[str]]]:
    """Generate cheatsheet data for all poolers.

    Returns:
        List of tuples containing (pooler_name, class_name, sparse, trainable, aux_loss, paper_links)
    """
    cheatsheet_data = []

    for pooler_name, pooler_class in pooler_map.items():
        class_name = pooler_class.__name__

        # Check properties
        sparse = supports_sparse(pooler_name)
        trainable = supports_trainable(pooler_name)
        aux_loss = supports_aux_loss(pooler_name)
        paper_links = extract_paper_links(pooler_class)

        cheatsheet_data.append(
            (pooler_name, class_name, sparse, trainable, aux_loss, paper_links)
        )

    # Sort by class name for consistent ordering
    cheatsheet_data.sort(key=lambda x: x[1])

    return cheatsheet_data


def print_cheatsheet() -> None:
    """Print the pooler cheatsheet in a readable format."""
    data = get_pooler_cheatsheet()

    print("TGP Pooler Cheatsheet")
    print("=" * 80)
    print(
        f"{'Name':<12} {'Class':<25} {'Sparse':<8} {'Trainable':<10} {'Aux Loss':<10} {'Paper Links':<15}"
    )
    print("-" * 80)

    for pooler_name, class_name, sparse, trainable, aux_loss, paper_links in data:
        sparse_mark = "✓" if sparse else ""
        trainable_mark = "✓" if trainable else ""
        aux_loss_mark = "✓" if aux_loss else ""
        paper_links_str = ", ".join(paper_links) if paper_links else ""

        print(
            f"{pooler_name:<12} {class_name:<25} {sparse_mark:<8} {trainable_mark:<10} {aux_loss_mark:<10} {paper_links_str:<15}"
        )


if __name__ == "__main__":
    print_cheatsheet()
