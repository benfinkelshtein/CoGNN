from enum import Enum, auto
from torch.nn import Module
from typing import List, Callable

from models.layers import WeightedGCNConv, WeightedGINConv, WeightedGNNConv, GraphLinear


class ModelType(Enum):
    """
        an object for the different core
    """
    GCN = auto()
    GIN = auto()
    LIN = auto()

    SUM_GNN = auto()
    MEAN_GNN = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()

    def load_component_cls(self):
        if self is ModelType.GCN:
            return WeightedGCNConv
        elif self is ModelType.GIN:
            return WeightedGINConv
        elif self in [ModelType.SUM_GNN, ModelType.MEAN_GNN]:
            return WeightedGNNConv
        elif self is ModelType.LIN:
            return GraphLinear
        else:
            raise ValueError(f'model {self.name} not supported')

    def is_gcn(self):
        return self is ModelType.GCN

    def get_component_list(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, bias: bool,
                           edges_required: bool, gin_mlp_func: Callable) -> List[Module]:
        dim_list = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        if self is ModelType.GCN:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, bias=bias)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.GIN:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, bias=bias,
                                                        mlp_func=gin_mlp_func)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self in [ModelType.SUM_GNN, ModelType.MEAN_GNN]:
            aggr = 'mean' if self is ModelType.MEAN_GNN else 'sum'
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, aggr=aggr,
                                                        bias=bias)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.LIN:
            assert not edges_required, f'env does not support {self.name}'
            component_list = \
                [self.load_component_cls()(in_features=in_dim_i, out_features=out_dim_i, bias=bias)
                 for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        else:
            raise ValueError(f'model {self.name} not supported')
        return component_list
