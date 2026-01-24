import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv, GCNConv

from tgp.poolers import get_pooler, pooler_map

seed_everything(8)  # Reproducibility

for POOLER, value in pooler_map.items():  # Use all poolers
    # for POOLER in ['mincut']:                 # Test a specific pooler

    pooler_cls = pooler_map[POOLER]
    print(f"Using pooler: {POOLER}")

    if POOLER == "pan":
        pass
    else:
        ### Get the data
        dataset = TUDataset(
            root="../data/TUDataset",
            name="MUTAG",
            pre_transform=pooler_cls.data_transforms(),
            force_reload=True,
        )
        train_loader = DataLoader(dataset[:0.9], batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset[0.9:], batch_size=32)

        PARAMS = {
            "cached": False,
            "lift": "inverse",
            "s_inv_op": "transpose",
            "reduce_red_op": "mean",
            "connect_red_op": "mean",
            "loss_coeff": 1.0,
            "k": dataset._data.num_nodes // len(dataset) // 2,
            "order_k": 2,
            "cache_sel": False,
            "cache_conn": False,
            "ratio": 0.25,
            "remove_self_loops": True,
            "scorer": "degree",
            "adj_transpose": True,
        }

        ### Model definition
        class Net(torch.nn.Module):
            def __init__(
                self, hidden_channels=64, pooler_type=POOLER, pooler_kwargs=PARAMS
            ):
                super().__init__()

                num_features = dataset.num_features
                num_classes = dataset.num_classes

                # First MP layer
                self.conv1 = GCNConv(
                    in_channels=num_features, out_channels=hidden_channels
                )

                # Pooling
                pooler_kwargs["in_channels"] = hidden_channels
                self.pooler = get_pooler(pooler_type, **pooler_kwargs)
                print(self.pooler)

                # Second MP layer
                if self.pooler.block_diags_output:
                    self.conv2 = GCNConv(
                        in_channels=hidden_channels, out_channels=hidden_channels
                    )
                else:
                    self.conv2 = DenseGCNConv(
                        in_channels=hidden_channels, out_channels=hidden_channels
                    )

                # Readout layer
                self.lin = torch.nn.Linear(hidden_channels, num_classes)

            def forward(self, x, edge_index, edge_weight, batch=None):
                # First MP layer
                x = self.conv1(x, edge_index, edge_weight)
                x = F.relu(x)

                # Pooling
                out = self.pooler(
                    x=x, adj=edge_index, edge_weight=edge_weight, batch=batch
                )
                x_pool, adj_pool = out.x, out.edge_index

                # Second MP layer
                if self.pooler.block_diags_output:
                    x = self.conv2(x_pool, adj_pool, out.edge_weight)
                else:
                    x = self.conv2(x_pool, adj_pool)
                x = F.relu(x)

                # Global pooling
                x = self.pooler.global_pool(x, reduce_op="sum", batch=out.batch)

                # Readout layer
                x = self.lin(x)

                if out.loss is not None:
                    return F.log_softmax(x, dim=-1), sum(out.get_loss_value())
                else:
                    return F.log_softmax(x, dim=-1), torch.tensor(0.0)

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
                output, aux_loss = model(
                    data.x, data.edge_index, data.edge_weight, data.batch
                )
                loss = F.nll_loss(output, data.y.view(-1)) + aux_loss
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
                pred = model(data.x, data.edge_index, data.edge_weight, data.batch)[
                    0
                ].argmax(dim=-1)
                correct += int(pred.eq(data.y.view(-1)).sum())
            return correct / len(loader.dataset)

        ### Training loop
        best_val_acc = test_acc = 0
        for epoch in range(1, 11):
            train_loss = train()
            val_acc = test(test_loader)
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
