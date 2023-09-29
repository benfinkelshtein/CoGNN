from torch import Tensor, nn
from torch_geometric.typing import Adj
from torch.nn import Module, ModuleList

from helpers.classes import GumbelArgs


class TempSoftPlus(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_dim: int):
        super(TempSoftPlus, self).__init__()
        model_list =\
            gumbel_args.temp_model_type.get_component_list(in_dim=env_dim, hidden_dim=env_dim, out_dim=1, num_layers=1,
                                                           bias=False, edges_required=False,
                                                           gin_mlp_func=gumbel_args.gin_mlp_func)
        self.linear_model = ModuleList(model_list)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = gumbel_args.tau0

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        x = self.linear_model[0](x=x, edge_index=edge_index,edge_attr = edge_attr)
        x = self.softplus(x) + self.tau0
        temp = x.pow_(-1)
        return temp.masked_fill_(temp == float('inf'), 0.)
