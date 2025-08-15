import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import ARMAConv

from tgp.data import PoolDataLoader, PreCoarsening
from tgp.poolers import get_pooler
from tgp.reduce import BaseReduce, global_reduce

seed_everything(8)

N_LEVELS = 2  # Number of coarsening levels
poolers = {
    "lap": {},
    "ndp": {},
    "nmf": {"k": 5},
    "graclus": {},
    "kmis": {"scorer": "degree", "order_k": 2},
}

for _pool, args in poolers.items():
    pooler = get_pooler(_pool, **args)

    ### Get the data
    dataset = TUDataset(
        root="../data/TUDataset",
        name="MUTAG",
        pre_transform=PreCoarsening(
            pooler=pooler,
            recursive_depth=N_LEVELS,
        ),
        force_reload=True,
    )
    train_loader = PoolDataLoader(dataset[:0.9], batch_size=32, shuffle=True)
    test_loader = PoolDataLoader(dataset[0.9:], batch_size=32)

    print(dataset[0])
    next_batch = next(iter(train_loader))
    print(next_batch)
    print(next_batch.pooled_data[0])

    ### Model definition
    class Net(torch.nn.Module):
        def __init__(self, hidden_channels=64, reducer=BaseReduce()):
            super().__init__()

            num_features = dataset.num_features
            num_classes = dataset.num_classes

            # First MP layer
            self.conv1 = ARMAConv(
                in_channels=num_features, out_channels=hidden_channels, num_layers=2
            )

            # Pooling
            self.reducer = reducer

            # Second MP layer
            self.next_conv = torch.nn.ModuleList(
                [
                    ARMAConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        num_layers=2,
                    )
                    for _ in range(N_LEVELS)
                ]
            )

            # Readout layer
            self.lin = torch.nn.Linear(hidden_channels, num_classes)

        def forward(self, data):
            # First MP layer
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.relu(x)

            # Pooling
            for pooled, conv in zip(data.pooled_data, self.next_conv):
                x, _ = self.reducer(x=x, so=pooled.so)

                # Next MP layer
                x = conv(x, pooled.edge_index, pooled.edge_weight)
                x = F.relu(x)

            # Global pooling
            x = global_reduce(x, reduce_op="sum", batch=pooled.batch)

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
    for epoch in range(1, 11):
        train_loss = train()
        val_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
