import os.path as osp

import torch
import torch_geometric.transforms as T
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch.nn import ModuleList
from torch_geometric import seed_everything, utils
from torch_geometric.datasets import Planetoid

from tgp.mp import GTVConv
from tgp.poolers import AsymCheegerCutPooling

seed_everything(8)

### Get the data
dataset = "cora"
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


### Model definition
class Net(torch.nn.Module):
    def __init__(
        self,
        mp_units,
        mp_act,
        in_channels,
        n_clusters,
        mlp_units=[],
        mlp_act="Identity",
    ):
        super().__init__()

        # Message passing layers
        mp = [GTVConv(in_channels, mp_units[0], act=mp_act, delta_coeff=0.311)]
        for i in range(len(mp_units) - 1):
            mp.append(
                GTVConv(mp_units[i], mp_units[i + 1], act=mp_act, delta_coeff=0.311)
            )
        self.mp = ModuleList(mp)
        out_chan = mp_units[-1]

        # Pooling
        self.pooler = AsymCheegerCutPooling(
            in_channels=[out_chan] + mlp_units,
            k=n_clusters,
            totvar_coeff=0.785,
            balance_coeff=0.514,
            act=mlp_act,
        )

    def forward(self, x, edge_index, edge_weight):
        for i in range(len(self.mp)):
            x = self.mp[i](x, edge_index, edge_weight)

        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)

        out = self.pooler(x=x, adj=adj)
        s = out.so.s[0]
        aux_loss = sum(out.get_loss_value())

        return s, aux_loss


### Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
model = Net(
    mp_units=[512] * 2,
    mp_act="elu",
    in_channels=dataset.num_features,
    n_clusters=dataset.num_classes,
    mlp_units=[256],
    mlp_act="relu",
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train():
    model.train()
    optimizer.zero_grad()
    _, loss = model(data.x, data.edge_index, data.edge_weight)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    clust, _ = model(data.x, data.edge_index, data.edge_weight)
    return NMI(clust.max(1)[1].cpu(), data.y.cpu())


### Training loop
for epoch in range(1, 1001):
    train_loss = train()
    nmi = test()
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}")
