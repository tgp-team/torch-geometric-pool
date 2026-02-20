"""Graph classification with poolers whose reduce step is replaced by AggrReduce.

This example instantiates MinCut, TopK, and Graclus poolers and swaps their
reducer for AggrReduce with different PyG aggregators (Sum, Mean, LSTM, Set2Set).
Graph-level readout uses the same aggregator for consistency.
"""

import time

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv, GCNConv

from tgp.poolers import get_pooler, pooler_map
from tgp.reduce import AggrReduce, get_aggr, readout

seed_everything(8)

# Poolers and aggregators to try
# POOLER_NAMES = ["topk", "lap", "mincut", "graclus"]
POOLER_NAMES = ["lap"]
# AGGR_NAMES = ["sum", "mean", "lstm", "set2set"]
AGGR_NAMES = ["mean"]


def readout_dim_for_aggr(aggr_name: str, in_channels: int) -> int:
    """Output feature dim after readout for the given aggregator (e.g. Set2Set doubles it)."""
    if aggr_name == "set2set":
        return 2 * in_channels
    return in_channels


def run_pooler_aggr(pooler_name: str, aggr_name: str, hidden_channels: int = 64):
    """Train a small classifier for one (pooler, aggregator) pair."""
    pooler_cls = pooler_map[pooler_name]
    dataset = TUDataset(
        root="../data/TUDataset",
        name="MUTAG",
        pre_transform=pooler_cls.data_transforms(),
        force_reload=True,
    )
    train_loader = DataLoader(
        dataset[: int(0.9 * len(dataset))], batch_size=32, shuffle=True
    )
    test_loader = DataLoader(dataset[int(0.9 * len(dataset)) :], batch_size=32)

    # Pooler kwargs (same spirit as classification.py)
    k = max(1, dataset._data.num_nodes // len(dataset) // 2)
    if pooler_name == "mincut":
        pooler_kwargs = {
            "in_channels": hidden_channels,
            "k": k,
            "cached": False,
            "lift": "inverse",
            "s_inv_op": "transpose",
            "remove_self_loops": True,
            "adj_transpose": True,
            "sparse_output": False,
            "batched": True,
        }
    elif pooler_name == "topk":
        pooler_kwargs = {
            "in_channels": hidden_channels,
            "ratio": 0.25,
            "cached": False,
            "s_inv_op": "transpose",
            "remove_self_loops": True,
        }
    else:  # graclus
        pooler_kwargs = {
            "cached": False,
            "s_inv_op": "transpose",
            "remove_self_loops": True,
        }

    pooler = get_pooler(pooler_name, **pooler_kwargs)

    aggr_kwargs = {"in_channels": hidden_channels}
    if aggr_name == "set2set":
        aggr_kwargs["processing_steps"] = 3
    pooler.reducer = AggrReduce(get_aggr(aggr_name, **aggr_kwargs))

    readout_dim = readout_dim_for_aggr(aggr_name, hidden_channels)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    use_dense_pool_adj = getattr(pooler, "is_dense", False) and not getattr(
        pooler, "sparse_output", True
    )
    pool_hidden = getattr(pooler, "num_modes", 1) * readout_dim

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.pooler = pooler
            if use_dense_pool_adj:
                self.conv2 = DenseGCNConv(pool_hidden, hidden_channels)
            else:
                self.conv2 = GCNConv(pool_hidden, hidden_channels)
            self.lin = torch.nn.Linear(readout_dim, num_classes)

        def forward(self, x, edge_index, edge_weight, batch=None):
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            out = self.pooler(x=x, adj=edge_index, edge_weight=edge_weight, batch=batch)
            x_pool, adj_pool = out.x, out.edge_index
            mask_pool = getattr(out, "mask", None)
            if use_dense_pool_adj:
                x = F.relu(self.conv2(x_pool, adj_pool, mask=mask_pool))
            else:
                x = F.relu(self.conv2(x_pool, adj_pool, out.edge_weight))
            # Readout: mask only for dense x (3D)
            readout_mask = mask_pool if (x.dim() == 3) else None
            x = readout(
                x,
                reduce_op=aggr_name,
                batch=out.batch,
                mask=readout_mask,
                **aggr_kwargs,
            )
            x = self.lin(x)
            aux = (
                sum(out.get_loss_value()) if out.loss is not None else torch.tensor(0.0)
            )
            return F.log_softmax(x, dim=-1), aux

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    # Ensure pooler's reducer (and its aggr submodule) are on the same device
    if hasattr(model.pooler, "reducer") and model.pooler.reducer is not None:
        model.pooler.reducer.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train_epoch():
        model.train()
        total = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, aux = model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = F.nll_loss(out, data.y.view(-1)) + aux
            loss.backward()
            total += data.y.size(0) * float(loss)
            optimizer.step()
        return total / len(dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_weight, data.batch)[
                0
            ].argmax(dim=-1)
            correct += int(pred.eq(data.y.view(-1)).sum())
        return correct / len(loader.dataset)

    # Short run: 5 epochs
    for epoch in range(1, 6):
        train_loss = train_epoch()
        acc = test(test_loader)
        print(f"  Epoch {epoch:02d}  loss={train_loss:.4f}  test_acc={acc:.4f}")
    return test(test_loader)


if __name__ == "__main__":
    print("Poolers: AggrReduce with Sum, Mean, LSTM, Set2Set\n")
    for pooler_name in POOLER_NAMES:
        for aggr_name in AGGR_NAMES:
            print(f"--- {pooler_name} + {aggr_name} ---")
            try:
                t0 = time.perf_counter()
                acc = run_pooler_aggr(pooler_name, aggr_name)
                elapsed = time.perf_counter() - t0
                print(f"  Final test acc: {acc:.4f}  time: {elapsed:.1f}s\n")
            except Exception as e:
                print(f"  Failed: {e}\n")
