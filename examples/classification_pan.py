import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, PANConv
from torch_sparse import SparseTensor

from tgp.poolers import get_pooler, pooler_map

POOLER = "pan"
pooler_cls = pooler_map[POOLER]
print(f"Using pooler: {POOLER}")

seed_everything(8)

### Get the data
dataset = TUDataset(
    root="../data/TUDataset", name="MUTAG", pre_transform=pooler_cls.data_transforms()
)
train_loader = DataLoader(dataset[:0.9], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[0.9:], batch_size=32)

PARAMS = {
    "cached": False,
    "s_inv_op": "transpose",
    "cache_sel": False,
    "cache_conn": False,
    "ratio": 0.25,
    "reduce_red_op": "mean",
    "connect_red_op": "mean",
    "lift_red_op": "mean",
    "multiplier": 1.0,
}


### Model definition
class Net(torch.nn.Module):
    def __init__(self, hidden_channels=64, pooler_type=POOLER, pooler_kwargs=PARAMS):
        super().__init__()

        num_features = dataset.num_features
        num_classes = dataset.num_classes

        # First MP layer
        self.conv1 = PANConv(
            in_channels=num_features, out_channels=hidden_channels, filter_size=3
        )

        # Pooling
        self.pooler = pooler_kwargs.update({"in_channels": hidden_channels})
        self.pooler = get_pooler(pooler_type, **pooler_kwargs)
        print(self.pooler)

        # Second MP layer
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)

        # Readout layer
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch=None):
        edge_index = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)

        # First MP layer
        x, M = self.conv1(x, edge_index)
        x = F.relu(x)

        # Pooling
        out = self.pooler(x=x, adj=M, batch=batch)
        x_pool, adj_pool = out.x, out.edge_index

        # Second MP layer
        x = self.conv2(x_pool, adj_pool)
        x = F.relu(x)

        # Global pooling
        x = self.pooler.global_pool(x, reduce_op="sum", batch=out.batch)

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
        output = model(data.x, data.edge_index, data.edge_weight, data.batch)
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
        pred = model(data.x, data.edge_index, data.edge_weight, data.batch)[0].argmax(
            dim=-1
        )
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


### Training loop
best_val_acc = test_acc = 0
for epoch in range(1, 11):
    train_loss = train()
    val_acc = test(test_loader)
    print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Val Acc: {val_acc:.3f}")
