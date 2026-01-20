import os.path as osp

import torch
import torch_geometric.transforms as T
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch.nn import ModuleList
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ARMAConv

from tgp.poolers import get_pooler, pooler_map

seed_everything(8)

poolers = ["acc", "spbnpool", "bnpool", "diff", "dmon", "hosc", "jb", "mincut"]
for POOLER in poolers:
    pooler_cls = pooler_map[POOLER]
    print(f"Using pooler: {POOLER}")

    ### Get the data
    dataset = "cora"
    trans = (
        T.NormalizeFeatures()
        if pooler_cls.data_transforms() is None
        else T.Compose([T.NormalizeFeatures(), pooler_cls.data_transforms()])
    )
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", dataset)
    dataset = Planetoid(path, dataset, transform=trans)
    data = dataset[0]

    PARAMS = {
        "loss_coeff": 1.0,
        "k": dataset.num_classes,
        "normalize_loss": True,
        "adj_transpose": True,
        "cache_preprocessing": True,
        "in_channels": [16],
        "act": "ReLU",
    }

    ### Model definition
    class Net(torch.nn.Module):
        def __init__(
            self,
            mp_units,
            mp_act,
            in_channels,
            pooler_type=POOLER,
            pooler_kwargs=PARAMS,
        ):
            super().__init__()

            mp_act = getattr(torch.nn, mp_act)(inplace=True)

            # Message passing layers
            mp = [
                ARMAConv(in_channels, mp_units[0], num_layers=2),
                mp_act,
            ]
            for i in range(len(mp_units) - 1):
                mp.append(ARMAConv(mp_units[i], mp_units[i + 1], num_layers=2))
                mp.append(mp_act)
            self.mp = ModuleList(mp)
            out_chan = mp_units[-1]

            # Pooling
            pooler_kwargs.update(
                {"in_channels": [out_chan] + pooler_kwargs["in_channels"]}
            )
            self.pooler = get_pooler(pooler_type, **pooler_kwargs)
            print(self.pooler)

        def forward(self, x, edge_index, edge_weight):
            for i in range(len(self.mp)):
                if i % 2 == 0:
                    x = self.mp[i](x, edge_index, edge_weight)
                else:
                    x = self.mp[i](x)

            out = self.pooler(x=x, adj=edge_index, edge_weight=edge_weight)
            s_out = out.so.s
            # check if s_out has batch dimension
            if s_out.dim() == 3:
                s_out = s_out[0]
            aux_loss = sum(out.get_loss_value())

            return s_out, aux_loss

    ### Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = Net(mp_units=[64] * 2, mp_act="ReLU", in_channels=dataset.num_features).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        _, loss = model(data.x, data.edge_index, data.edge_weight)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        clust, _ = model(data.x, data.edge_index, data.edge_weight)
        return NMI(clust.max(1)[1].cpu(), data.y.cpu())

    ### Training loop
    for epoch in range(1, 11):
        train_loss = train()
        nmi = test()
        print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}")
