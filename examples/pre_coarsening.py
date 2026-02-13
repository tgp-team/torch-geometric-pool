import time

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import ARMAConv

from tgp.data import PoolDataLoader, PreCoarsening
from tgp.reduce import global_reduce

seed_everything(8)

pooling_schedules = {
    "nopool->nopool": ["nopool", "nopool"],
    "ndp->ndp": ["ndp", "ndp"],
    "graclus->graclus": ["graclus", "graclus"],
    "kmis->kmis": [
        ("kmis", {"scorer": "degree", "order_k": 2}),
        ("kmis", {"scorer": "degree", "order_k": 2}),
    ],
    # Poolers with different parameters across levels
    "nmf->nmf": [("nmf", {"k": 8}), ("nmf", {"k": 4})],
    "eigen->eigen": [
        ("eigen", {"k": 5, "num_modes": 3}),
        ("eigen", {"k": 3, "num_modes": 3}),
    ],
    # Mixed poolers
    "ndp->eigen": [
        "ndp",
        ("eigen", {"k": 4, "num_modes": 3}),
    ],
}

for schedule_name, level_specs in pooling_schedules.items():
    pre_transform = PreCoarsening(poolers=level_specs)
    level_poolers = pre_transform.poolers
    num_levels = len(level_poolers)

    print(f"=== Using schedule: {schedule_name} ({num_levels} levels) ===")

    ### Get the data
    dataset = TUDataset(
        root="../data/TUDataset",
        name="MUTAG",
        pre_transform=pre_transform,
        force_reload=True,
    )
    train_loader = PoolDataLoader(dataset[:0.9], batch_size=32, shuffle=True)
    test_loader = PoolDataLoader(dataset[0.9:], batch_size=32)

    print(dataset[0])
    next_batch = next(iter(train_loader))
    print(next_batch)
    print(next_batch.pooled_data[0])

    # EigenPooling expands features: [K, H*d]; others use num_modes=1
    level_num_modes = [getattr(pooler, "num_modes", 1) for pooler in level_poolers]

    ### Model definition
    class Net(torch.nn.Module):
        def __init__(self, hidden_channels=64):
            super().__init__()

            num_features = dataset.num_features
            num_classes = dataset.num_classes

            # First MP layer
            self.conv1 = ARMAConv(
                in_channels=num_features, out_channels=hidden_channels, num_layers=2
            )

            # Pooling
            self.reducers = torch.nn.ModuleList(
                [pooler.reducer for pooler in level_poolers]
            )

            # Second MP layer
            self.next_conv = torch.nn.ModuleList()
            for num_modes in level_num_modes:
                in_ch = hidden_channels * num_modes
                self.next_conv.append(
                    ARMAConv(
                        in_channels=in_ch,
                        out_channels=hidden_channels,
                        num_layers=2,
                    )
                )

            # Readout layer
            self.lin = torch.nn.Linear(hidden_channels, num_classes)

        def forward(self, data):
            # First MP layer
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.relu(x)

            # Pooling
            for pooled, conv, reducer in zip(
                data.pooled_data, self.next_conv, self.reducers
            ):
                x, _ = reducer(x=x, so=pooled.so)

                # Next MP layer
                x = conv(x, pooled.edge_index, pooled.edge_weight)
                x = F.relu(x)

            # Global pooling
            x = global_reduce(
                x,
                reduce_op="sum",
                batch=pooled.batch,
                mask=getattr(pooled, "mask", None),
            )

            # Readout layer
            x = self.lin(x)

            return F.log_softmax(x, dim=-1)

    ### Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train():
        model.train()
        loss_all = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y.view(-1))
            loss.backward()
            loss_all += data.y.size(0) * float(loss)
            optimizer.step()
        return loss_all / len(dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data).argmax(dim=-1)
            correct += int(pred.eq(data.y.view(-1)).sum())
        return correct / len(loader.dataset)

    ### Training loop
    best_val_acc = test_acc = 0
    start_time = time.time()
    for epoch in range(1, 11):
        train_loss = train()
        val_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
