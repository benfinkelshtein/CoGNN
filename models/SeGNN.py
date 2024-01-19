import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np

from helpers.classes import EigenArgs, GumbelArgs, EnvArgs, ActionNetArgs, Pool, DataSetEncoders
from models.temp import TempSoftPlus
from models.action import ActionNet


class DenseSEGNN(Module):
    def __init__(self, eigen_args: EigenArgs, env_args: EnvArgs, pool: Pool):
        super(DenseSEGNN, self).__init__()
        self.env_args = env_args

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        self.eigen_net = eigen_args.load_net()
        self.use_encoders = env_args.dataset_encoders.use_encoders()

        self.eigen_dim = eigen_args.eigen_dim

        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()

        # Encoder types
        self.dataset_encoder = env_args.dataset_encoders
        self.env_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args.env_dim, model_type=env_args.model_type)

        # Pooling function to generate whole-graph embeddings
        self.pooling = pool.get()

    def forward(self, x: Tensor, edge_index: Adj, pestat, Q, edge_attr: OptTensor = None, batch: OptTensor = None,
                edge_ratio_node_mask: OptTensor = None, ) -> Tuple[Tensor, Tensor]:
        result = 0

        calc_stats = edge_ratio_node_mask is not None
        if calc_stats:
            edge_ratio_edge_mask = edge_ratio_node_mask[edge_index[0]] & edge_ratio_node_mask[edge_index[1]]
            edge_ratio_list = []

        # bond encode
        if edge_attr is None or self.env_bond_encoder is None:
            env_edge_embedding = None
        else:
            env_edge_embedding = self.env_bond_encoder(edge_attr)


        # node encode  
        x = self.env_net[0](x, pestat)  # (N, F) encoder
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            # environment
            hx = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index, edge_attr=env_edge_embedding)
            hx = self.dropout(hx)
            hx = hx

            # eigen
            he = self.eigen_net[1 + gnn_idx](x=x, q=Q)
            #he = he
            he = self.dropout(he)

            h = hx + he 
            h = h/2
            if self.skip:
                x = x + h
            else:
                x = h

        x = self.hidden_layer_norm(x)
        x = self.pooling(x, batch=batch)
        x = self.env_net[-1](x)  # decoder
        result = result + x

        return result