import os.path as osp

import torch
import torch_geometric.transforms as T
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch.nn import ModuleList
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from tgp.poolers import get_pooler, pooler_map

seed_everything(8)

poolers = ["mincut", "diff", "jb", "acc", "dmon", "hosc", "bnpool"]
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
        "in_channels": [16],
        "act": "ReLU",
    }

    if POOLER == "bnpool":
        PARAMS["max_k"] = 20
        PARAMS.pop('k')

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
                GCNConv(in_channels, mp_units[0], normalize=False, cached=False),
                mp_act,
            ]
            for i in range(len(mp_units) - 1):
                mp.append(
                    GCNConv(mp_units[i], mp_units[i + 1], normalize=False, cached=False)
                )
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

            _, adj, _ = self.pooler.preprocessing(
                x=x, edge_index=edge_index, edge_attr=edge_weight, use_cache=True
            )

            out = self.pooler(x=x, adj=adj)
            s = out.so.s[0]
            aux_loss = sum(out.get_loss_value())

            return s, aux_loss

    ### Model setup
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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
