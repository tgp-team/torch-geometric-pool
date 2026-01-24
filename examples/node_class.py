import time

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DenseGCNConv, GCNConv

from tgp.poolers import get_pooler, pooler_map

seed_everything(8)

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]

for POOLER, value in pooler_map.items():  # Use all poolers
    # for POOLER in ["bnpool_u"]:  # Test a specific pooler
    print(f"Using pooler: {POOLER}")

    if POOLER == "pan":
        pass
    else:
        PARAMS = {
            "cached": True,
            "cache_preprocessing": True,
            "lift": "precomputed",
            "s_inv_op": "transpose",
            "lift_red_op": "mean",
            "loss_coeff": 10.0,
            "k": data.num_nodes // 20,
            "order_k": 2,
            "ratio": 0.25,
            "remove_self_loops": True,
            "scorer": "degree",
            "reduce": "sum",
            "edge_weight_norm": False,
            "degree_norm": True,
            "block_diags_output": False,
        }

        #### Model definition
        class Net(torch.nn.Module):
            def __init__(
                self,
                hidden_channels,
                dropout=0.5,
                pooler_type=POOLER,
                pooler_kwargs=PARAMS,
            ):
                super().__init__()

                self.conv_enc = GCNConv(dataset.num_features, hidden_channels)

                pooler_kwargs["in_channels"] = hidden_channels
                self.pooler = get_pooler(pooler_type, **pooler_kwargs)
                print(self.pooler)
                self.pooler.reset_parameters()

                self.use_dense_pool_adj = (
                    self.pooler.is_dense and not self.pooler.block_diags_output
                )
                if self.use_dense_pool_adj:
                    self.conv_pool = DenseGCNConv(hidden_channels, hidden_channels // 2)
                else:
                    self.conv_pool = GCNConv(hidden_channels, hidden_channels // 2)

                self.use_dense_input_adj = self.pooler.batched and getattr(
                    self.pooler, "cache_preprocessing", False
                )
                if self.use_dense_input_adj:
                    self.conv_dec = DenseGCNConv(
                        hidden_channels // 2, dataset.num_classes
                    )
                else:
                    self.conv_dec = GCNConv(hidden_channels // 2, dataset.num_classes)

                self.dropout = dropout

            def forward(self, x, edge_index, edge_weight, batch):
                # Encoder
                x = self.conv_enc(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # Pooling
                out = self.pooler(
                    x=x,
                    adj=edge_index,
                    edge_weight=edge_weight,
                    batch=batch,
                )
                x_pool, adj_pool = (
                    out.x,
                    out.edge_index,
                )

                # Bottleneck
                if self.use_dense_pool_adj:
                    x_pool = self.conv_pool(x_pool, adj_pool)
                else:
                    x_pool = self.conv_pool(x_pool, adj_pool, out.edge_weight)
                x_pool = F.relu(x_pool)
                x_pool = F.dropout(x_pool, p=self.dropout, training=self.training)

                # Decoder
                x_lift = self.pooler(
                    x=x_pool,
                    so=out.so,
                    lifting=True,
                    batch=batch,
                    batch_pooled=out.batch,
                )
                if self.use_dense_input_adj:
                    adj_dense = self.pooler.preprocessing_cache
                    if adj_dense is None:
                        raise RuntimeError(
                            "Dense decoder requires cache_preprocessing=True "
                            "to reuse the dense adjacency."
                        )
                    x = self.conv_dec(x_lift, adj_dense)
                else:
                    if x_lift.dim() == 3:
                        x_lift = x_lift[0]
                    x = self.conv_dec(x_lift, edge_index, edge_weight)

                if self.pooler.batched and x.dim() == 3:
                    x = x[0]
                if out.loss is not None:
                    return F.log_softmax(x, dim=-1), sum(out.get_loss_value())
                else:
                    return F.log_softmax(x, dim=-1), torch.tensor(0.0)

        ### Model setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net(hidden_channels=16).to(device)
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

        ### Training loop
        start_time = time.time()
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
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
