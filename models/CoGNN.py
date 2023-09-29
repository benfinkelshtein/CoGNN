import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np

from helpers.classes import GumbelArgs, EnvArgs, ActionNetArgs, Pool, DataSetEncoders
from models.temp import TempSoftPlus
from models.action import ActionNet


class CoGNN(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs, pool: Pool):
        super(CoGNN, self).__init__()
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        self.use_encoders = env_args.dataset_encoders.use_encoders()

        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()
        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)

        # Encoder types
        self.dataset_encoder = env_args.dataset_encoders
        self.env_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args.env_dim, model_type=env_args.model_type)
        self.act_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=action_args.hidden_dim, model_type=action_args.model_type)

        # Pooling function to generate whole-graph embeddings
        self.pooling = pool.get()

    def forward(self, x: Tensor, edge_index: Adj, pestat, edge_attr: OptTensor = None, batch: OptTensor = None,
                edge_ratio_node_mask: OptTensor = None) -> Tuple[Tensor, Tensor]:
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
        if edge_attr is None or self.act_bond_encoder is None:
            act_edge_embedding = None
        else:
            act_edge_embedding = self.act_bond_encoder(edge_attr)

        # node encode  
        x = self.env_net[0](x, pestat)  # (N, F) encoder
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                        act_edge_attr=act_edge_embedding)  # (N, 2)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                          act_edge_attr=act_edge_embedding)  # (N, 2)

            temp = self.temp_model(x=x, edge_index=edge_index,
                                   edge_attr=env_edge_embedding) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])

            # environment
            out = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                            edge_attr=env_edge_embedding)
            out = self.dropout(out)
            out = self.act(out)

            if calc_stats:
                edge_ratio = edge_weight[edge_ratio_edge_mask].sum() / edge_weight[edge_ratio_edge_mask].shape[0]
                edge_ratio_list.append(edge_ratio.item())

            if self.skip:
                x = x + out
            else:
                x = out

        x = self.hidden_layer_norm(x)
        x = self.pooling(x, batch=batch)
        x = self.env_net[-1](x)  # decoder
        result = result + x

        if calc_stats:
            edge_ratio_tensor = torch.tensor(edge_ratio_list, device=x.device)
        else:
            edge_ratio_tensor = -1 * torch.ones(size=(self.num_layers,), device=x.device)
        return result, edge_ratio_tensor

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
