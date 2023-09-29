from enum import Enum, auto
from torch.nn import Linear, ModuleList, Module, Dropout, ReLU, GELU, Sequential
from torch import Tensor
from typing import NamedTuple, Any, Callable
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool, global_add_pool

from helpers.metrics import MetricType
from helpers.model import ModelType
from helpers.encoders import DataSetEncoders, PosEncoder
from lrgb.encoders.composition import Concat2NodeEncoder


class ActivationType(Enum):
    """
        an object for the different activation types
    """
    RELU = auto()
    GELU = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ActivationType[s]
        except KeyError:
            raise ValueError()

    def get(self):
        if self is ActivationType.RELU:
            return F.relu
        elif self is ActivationType.GELU:
            return F.gelu
        else:
            raise ValueError(f'ActivationType {self.name} not supported')

    def nn(self) -> Module:
        if self is ActivationType.RELU:
            return ReLU()
        elif self is ActivationType.GELU:
            return GELU()
        else:
            raise ValueError(f'ActivationType {self.name} not supported')


class GumbelArgs(NamedTuple):
    learn_temp: bool
    temp_model_type: ModelType
    tau0: float
    temp: float
    gin_mlp_func: Callable


class Pool(Enum):
    """
        an object for the different activation types
    """
    NONE = auto()
    MEAN = auto()
    SUM = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return Pool[s]
        except KeyError:
            raise ValueError()

    def get(self):
        if self is Pool.MEAN:
            return global_mean_pool
        elif self is Pool.SUM:
            return global_add_pool
        elif self is Pool.NONE:
            return BatchIdentity()
        else:
            raise ValueError(f'Pool {self.name} not supported')


class EnvArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    env_dim: int

    layer_norm: bool
    skip: bool
    batch_norm: bool
    dropout: float
    act_type: ActivationType
    dec_num_layers: int
    pos_enc: PosEncoder
    dataset_encoders: DataSetEncoders

    metric_type: MetricType
    in_dim: int
    out_dim: int

    gin_mlp_func: Callable

    def load_net(self) -> ModuleList:
        if self.pos_enc is PosEncoder.NONE:
            enc_list = [self.dataset_encoders.node_encoder(in_dim=self.in_dim, emb_dim=self.env_dim)]
        else:
            if self.dataset_encoders is DataSetEncoders.NONE:
                enc_list = [self.pos_enc.get(in_dim=self.in_dim, emb_dim=self.env_dim)]
            else:
                enc_list = [Concat2NodeEncoder(enc1_cls=self.dataset_encoders.node_encoder,
                                               enc2_cls=self.pos_enc.get,
                                               in_dim=self.in_dim, emb_dim=self.env_dim,
                                               enc2_dim_pe=self.pos_enc.DIM_PE())]

        component_list =\
            self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.env_dim,  out_dim=self.env_dim,
                                               num_layers=self.num_layers, bias=True, edges_required=True,
                                               gin_mlp_func=self.gin_mlp_func)

        if self.dec_num_layers > 1:
            mlp_list = (self.dec_num_layers - 1) * [Linear(self.env_dim, self.env_dim),
                                                    Dropout(self.dropout), self.act_type.nn()]
            mlp_list = mlp_list + [Linear(self.env_dim, self.out_dim)]
            dec_list = [Sequential(*mlp_list)]
        else:
            dec_list = [Linear(self.env_dim, self.out_dim)]

        return ModuleList(enc_list + component_list + dec_list)


class ActionNetArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    hidden_dim: int

    dropout: float
    act_type: ActivationType

    env_dim: int
    gin_mlp_func: Callable
    
    def load_net(self) -> ModuleList:
        net = self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.hidden_dim, out_dim=2,
                                                 num_layers=self.num_layers, bias=True, edges_required=False,
                                                 gin_mlp_func=self.gin_mlp_func)
        return ModuleList(net)


class BatchIdentity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return x
