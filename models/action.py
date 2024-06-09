import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

from helpers.classes import ActionNetArgs


class ActionNet(nn.Module):
    def __init__(self, action_args: ActionNetArgs):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = action_args.num_layers
        self.net = action_args.load_net()
        self.dropout = nn.Dropout(action_args.dropout)
        self.act = action_args.act_type.get()

    def forward(self, x: Tensor, edge_index: Adj, env_edge_attr: OptTensor, act_edge_attr: OptTensor) -> Tensor:
        edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = self.act(x)
        x = self.net[-1](x=x, edge_index=edge_index, edge_attr=edge_attrs[-1])
        return x
