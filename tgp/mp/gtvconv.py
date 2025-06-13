import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric import utils
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


def gtv_adj_weights(
    edge_index, edge_weight, num_nodes=None, flow="source_to_target", coeff=1.0
):
    fill_value = 0.0

    assert flow in ["source_to_target", "target_to_source"]

    edge_index, tmp_edge_weight = utils.add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes
    )
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

    # Compute degrees
    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)

    # Compute laplacian: L = D - A = -A + D
    edge_weight = -edge_weight
    edge_weight[row == col] += deg

    # Compute adjusted laplacian: L_adjusted = I - delta*L = -delta*L + I
    edge_weight *= -coeff
    edge_weight[row == col] += 1

    return edge_index, edge_weight


class GTVConv(MessagePassing):
    r"""The GTVConv layer from the paper `"Total Variation Graph Neural Networks"
    <https://arxiv.org/abs/2211.06218>`_ (Hansen & Bianchi, ICML 2023).

    Args:
        in_channels (int):
            Size of the input node features.
        out_channels (int):
            Size of each output node features.
        bias (bool):
            If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        delta_coeff (float):
            Step size for gradient descent of Graph Total Variation
            (default: :obj:`1.0`)
        eps (float):
            Small number used to numerically stabilize the computation of
            new adjacency weights. (default: :obj:`1e-3`)
        act (any):
            Activation function. Must be compatible with
            `~torch_geometric.nn.resolver`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        delta_coeff: float = 1.0,
        eps: float = 1e-3,
        act="relu",
    ):
        super().__init__(aggr="add", flow="target_to_source")
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.delta_coeff = delta_coeff
        self.eps = eps

        self.act = activation_resolver(act)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight)
        zeros(self.bias)

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, mask=None
    ) -> Tensor:
        # Update node features
        x = x @ self.weight

        # Check if a dense adjacency is provided
        if isinstance(edge_index, Tensor) and edge_index.size(-1) == edge_index.size(
            -2
        ):
            x = x.unsqueeze(0) if x.dim() == 2 else x
            edge_index = (
                edge_index.unsqueeze(0) if edge_index.dim() == 2 else edge_index
            )
            B, N, _ = edge_index.size()

            # Absolute differences between neighbouring nodes
            batch_idx, node_i, node_j = torch.nonzero(edge_index, as_tuple=True)
            abs_diff = torch.sum(
                torch.abs(x[batch_idx, node_i, :] - x[batch_idx, node_j, :]), dim=-1
            )  # shape [B, E]

            # Gamma matrix
            mod_adj = torch.clone(edge_index)
            mod_adj[batch_idx, node_i, node_j] /= torch.clamp(abs_diff, min=self.eps)

            # Compute Laplacian L=D-A
            deg = torch.sum(mod_adj, dim=-1)
            mod_adj = -mod_adj
            mod_adj[:, range(N), range(N)] += deg

            # Compute modified laplacian: L_adjusted = I - delta*L
            mod_adj = -self.delta_coeff * mod_adj
            mod_adj[:, range(N), range(N)] += 1

            out = torch.matmul(mod_adj, x)

            if self.bias is not None:
                out = out + self.bias

            if mask is not None:
                out = out * mask.view(B, N, 1).to(x.dtype)

        else:
            if isinstance(edge_index, SparseTensor):
                row, col, edge_weight = edge_index.coo()
                edge_index = torch.stack((row, col), dim=0)
            else:
                row, col = edge_index

            # Absolute differences between neighbouring nodes
            abs_diff = torch.abs(x[row, :] - x[col, :])  # shape [E, in_channels]
            abs_diff = abs_diff.sum(dim=1)  # shape [E, ]

            # Gamma matrix
            denom = torch.clamp(abs_diff, min=self.eps)
            gamma_vals = 1 / denom if edge_weight is None else edge_weight / denom

            # Laplacian L=D-A
            lap_index, lap_weight = utils.get_laplacian(edge_index, gamma_vals)

            # Modified laplacian: I-delta*L
            lap_weight *= -self.delta_coeff
            mod_lap_index, mod_lap_weight = utils.add_self_loops(
                lap_index, lap_weight, fill_value=1.0, num_nodes=x.size(0)
            )

            out = self.propagate(
                edge_index=mod_lap_index, x=x, edge_weight=mod_lap_weight, size=None
            )

            if self.bias is not None:
                out = out + self.bias

        return self.act(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
