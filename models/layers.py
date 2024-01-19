from typing import Callable
import torch
from torch import Tensor
from torch.nn import Linear, ReLU, BatchNorm1d, Module

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
from torch.nn import Parameter
import torch.nn as nn

def get_virtualnode_mlp(emb_dim: int) -> Module:
    return torch.nn.Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(),
                               Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU())


class MolConv(MessagePassing):
    def message(self, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor = None) -> Tensor:
        if edge_attr is None:
            if edge_weight is None:
                return x_j
            else:
                return edge_weight.view(-1, 1) * x_j
        else:
            if edge_weight is None:
                return x_j + edge_attr
            else:
                return edge_weight.view(-1, 1) * (x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class WeightedGCNConv(MolConv):
    def __init__(self, in_channels: int, out_channels: int, bias: bool, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = True
        self.improved = False

        self.lin = Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        edge_index = remove_self_loops(edge_index=edge_index)[0]
        _, edge_attr = add_remaining_self_loops(edge_index, edge_attr, fill_value=1, num_nodes=x.shape[0])

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim),
            self.improved, self.add_self_loops, self.flow, x.dtype)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, edge_attr=edge_attr)
        out = self.lin(out)
        return out


# GIN convolution along the graph structure
class WeightedGINConv(MolConv):
    def __init__(self, in_channels: int, out_channels: int, bias: bool, mlp_func: Callable):
        """
            emb_dim (int): node embedding dimensionality
        """
        super(WeightedGINConv, self).__init__(aggr="add")

        self.mlp = mlp_func(in_channels=in_channels, out_channels=out_channels, bias=bias)
        self.eps = torch.Tensor([0])
        self.eps = torch.nn.Parameter(self.eps)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, edge_attr=edge_attr)

        return self.mlp((1 + self.eps.to(x.device)) * x + out)


class WeightedGNNConv(MolConv):
    def __init__(self, in_channels: int, out_channels: int, aggr: str, bias: bool, **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, edge_attr=edge_attr)
        out = self.lin(torch.cat((x, out), dim=-1))
        return out


class GraphLinear(Linear):
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        return super().forward(x)

class SETresholdedConv(MolConv):    
    def __init__(self, in_channels: int, out_channels: int, eigval_nr : int, aggr: str, bias: bool, **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.eigval_nr = eigval_nr
        self.eps = Parameter(torch.zeros(eigval_nr))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, Q_edge_index: Adj, edge_attr: OptTensor = None, Q_edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(Q_edge_index, x=x, edge_weight=Q_edge_weight, size=None, edge_attr=edge_attr)
        out = self.eps * out
        out = self.lin(torch.cat((x, out), dim=-1))
        return out


class SEDenseConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eigen_dim : int, bias: bool, **kwargs):
        super().__init__()
    
    # epsilon
        self.eigen_dim = eigen_dim
        self.eps = []
        self.bias = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = []
        #for i in range(eigen_dim):
        #    self.eps.append(Parameter(torch.randn(1)))
        #    self.bias.append(Parameter(torch.empty(out_channels)))
        #    self.lins.append(Linear(2 * in_channels, out_channels, bias=bias))
        self.eps = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(eigen_dim)])
        self.bias = nn.ParameterList([nn.Parameter(torch.empty(out_channels)) for _ in range(eigen_dim)])
        self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=bias) for _ in range(eigen_dim)])
    
    def forward(self, x: Tensor, q: Tensor) -> Tensor:
        h = []
        for i in range(2): 
            out = self.lins[i](x)
            out = self.eps[i] * torch.matmul(q[i+1].unsqueeze(dim=1), torch.matmul(q[i+1].unsqueeze(dim=1).T, out))
            if self.bias is not None:
                out = out + self.bias[i]
                h.append(out)

        # sum over the features h
        H = torch.stack(h, dim=-1).mean(dim=-1)

        return H
