import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DenseGCNConv, GCNConv
from torch_sparse import SparseTensor

from tgp.poolers import get_pooler, pooler_map

seed_everything(8)

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]

for POOLER, value in pooler_map.items():
# for POOLER in ['mincut']: # Test a specific pooler

    print(f"Using pooler: {POOLER}")

    if POOLER == "pan":
        pass
    else:
        PARAMS = {
            "cached": True,
            "lift": "precomputed",
            "s_inv_op": "transpose",
            "lift_red_op": "mean",
            "loss_coeff": 10.0,
            "k": data.num_nodes // 20,
            "ratio": 0.25,
            "remove_self_loops": True,
            "scorer": "degree",
            "reduce": "sum",
        }

        class GCN(torch.nn.Module):
            def __init__(
                self,
                hidden_channels,
                dropout=0.5,
                pooler_type=POOLER,
                pooler_kwargs=PARAMS,
            ):
                super().__init__()

                self.conv_enc = GCNConv(dataset.num_features, hidden_channels)

                pooler_kwargs.update({"in_channels": hidden_channels})
                self.pooler = get_pooler(pooler_type, **pooler_kwargs)
                print(self.pooler)
                self.pooler.reset_parameters()

                if self.pooler.is_dense:
                    self.conv_pool = DenseGCNConv(hidden_channels, hidden_channels // 2)
                    self.conv_dec = DenseGCNConv(
                        hidden_channels // 2, dataset.num_classes
                    )
                else:
                    self.conv_pool = GCNConv(hidden_channels, hidden_channels // 2)
                    self.conv_dec = GCNConv(hidden_channels // 2, dataset.num_classes)

                self.dropout = dropout

            def forward(self, x, edge_index, edge_weight, batch):
                edge_index = SparseTensor.from_edge_index(
                    edge_index, edge_attr=edge_weight
                )  # Optional, debug

                # Encoder
                x = self.conv_enc(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # Pooling
                x, edge_index, mask = self.pooler.preprocessing(
                    edge_index=edge_index, x=x, batch=batch, use_cache=True
                )
                out = self.pooler(x=x, adj=edge_index, batch=batch, mask=mask)
                x_pool, adj_pool, _pooled_edge_weights = (
                    out.x,
                    out.edge_index,
                    out.edge_weight,
                )
                # print(f"Expressive: {out.so.expressive}")

                # Bottleneck
                x_pool = self.conv_pool(
                    x_pool, adj_pool
                )  # edge_weight=pooled_edge_weights
                x_pool = F.relu(x_pool)
                x_pool = F.dropout(x_pool, p=self.dropout, training=self.training)

                # Decoder
                x_lift = self.pooler(x=x_pool, so=out.so, lifting=True)
                x = self.conv_dec(x_lift, edge_index)

                if self.pooler.is_dense:
                    x = x[0]
                if out.loss is not None:
                    return F.log_softmax(x, dim=-1), sum(out.get_loss_value())
                else:
                    return F.log_softmax(x, dim=-1), torch.tensor(0.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GCN(hidden_channels=16).to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

        def train():
            model.train()
            optimizer.zero_grad()
            out, aux_loss = model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + aux_loss
            loss.backward()
            optimizer.step()
            return loss.item()

        @torch.no_grad()
        def test():
            model.eval()
            out, _ = model(data.x, data.edge_index, data.edge_weight, data.batch)
            preds = out.argmax(dim=1)

            # Evaluate on train, validation, and test splits
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                correct = preds[mask].eq(data.y[mask]).sum().item()
                acc = correct / mask.sum().item()
                accs.append(acc)
            return accs

        # Training loop
        for epoch in range(1, 11):
            loss = train()
            train_acc, val_acc, test_acc = test()
            print(
                f"Epoch: {epoch:03d}, "
                f"Loss: {loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"Test Acc: {test_acc:.4f}"
            )
