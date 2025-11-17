from typing import Union, Optional

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import MetaLayer, GAT
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.data import Data
import numpy as np
import json

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
        y = y(x, c) ⊙ x + β(x, c)
    where:
      - x:  [*, feat_dim]
      - c:  [*, cond_dim]
    The internal MLP must accept (feat_dim + cond_dim) → (2 * feat_dim).
    """
    def __init__(self, feat_dim: int, cond_dim: int,
                 hidden: int = 100, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim

        # Build an MLP that maps (feat_dim + cond_dim) → (2 * feat_dim).
        # We replace the old lazy get_mlp with an explicit version:
        self.net = self._build_mlp(input_dim=feat_dim + cond_dim,
                                   hidden_dim=hidden,
                                   output_dim=2 * feat_dim,
                                   n_layers=n_layers,
                                   dropout=dropout)
        self.act = nn.Sigmoid()

    def _build_mlp(self, input_dim: int, hidden_dim: int,
                   output_dim: int, n_layers: int, dropout: float):
        """
        Simple helper to build an MLP with n_layers:
          - if n_layers == 1:  Linear(input_dim → output_dim)
          - else:  [Linear(input_dim→hidden_dim), SELU, LayerNorm, Dropout] x (n_layers-1),
                   followed by Linear(hidden_dim→output_dim)
        """
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # first hidden block
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))

            # middle hidden blocks (if >2)
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SELU())
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Dropout(dropout))

            # final layer to output_dim
            layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [E_or_N, feat_dim]
        cond: [E_or_N, cond_dim]
        """
        h = torch.cat([x, cond], dim=-1)            # [E_or_N, feat_dim + cond_dim]
        gamma_beta = self.net(h)                    # [E_or_N, 2*feat_dim]
        gamma, beta = gamma_beta.chunk(2, dim=-1)   # each [E_or_N, feat_dim]
        #gamma = self.act(gamma)                     # gate in (0,1)
        return gamma * x + beta                     # [E_or_N, feat_dim]


####### EdgeFiLMModel: condition on (src, dst, dilution_scalar) only #######

class EdgeFiLMModel(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 node_dim: int,
                 global_dim: int,
                 hidden: int = 50,
                 n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        
        #cond_dim = 2 * node_dim + global_dim
        cond_dim = 2 * node_dim + 1
        
        self.film = FiLM(
            feat_dim = edge_dim,
            cond_dim = cond_dim,
            hidden   = hidden,
            n_layers = n_layers,
            dropout  = dropout
        )

    def forward(self,
                src: torch.Tensor,
                dst: torch.Tensor,
                edge_attr: torch.Tensor,
                u: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        src:       [E, node_dim]       node features for edge source
        dst:       [E, node_dim]       node features for edge destination
        edge_attr: [E, edge_dim]       original edge attributes
        u:         [B, global_dim]     global feature for each graph
        batch:     [E]                 which graph each edge belongs to
        """
        # cond = torch.cat([src, dst, u[batch]], dim=-1)  # [E, node_dim + node_dim + 1]
        d = u[batch, -1:].contiguous()                  # [E, 1]
        cond = torch.cat([src, dst, d], dim=-1)         # [E, 2*node_dim + 1]
        return self.film(edge_attr, cond)               # [E, edge_dim]


####### NodeAttnFiLM: condition on (node_feat, dilution_scalar) only #######

class NodeAttnFiLM(nn.Module):
    def __init__(self,
                 node_dim: int,
                 global_dim: int,
                 n_heads: int = 5,
                 hidden: int = 50,
                 dropout: float = 0.0,
                 edge_dim: Optional[int] = None):
        super().__init__()
        assert node_dim % n_heads == 0, "node_dim must be divisible by n_heads"

        # Use the new GAT signature:
        self.gat = GAT(in_channels     = node_dim,
                       hidden_channels = node_dim // n_heads,
                       out_channels    = node_dim,
                       num_layers      = 1,
                       heads           = n_heads,
                       concat          = True,
                       dropout         = dropout,
                       v2              = True,
                       edge_dim        = edge_dim,     # or None if not using edge_attr
                       add_self_loops  = True)

        
        self.film = FiLM(feat_dim = node_dim,
                         cond_dim = 1,
                         hidden   = hidden,
                         n_layers = 1, # 2,
                         dropout  = dropout)

        self.mlp_out = self._build_mlp(input_dim = node_dim,
                                       hidden_dim= hidden,
                                       output_dim= node_dim,
                                       n_layers = 1 , # 2
                                       dropout  = dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.drop  = nn.Dropout(dropout)

    def _build_mlp(self, input_dim: int, hidden_dim: int,
                   output_dim: int, n_layers: int, dropout: float):
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SELU())
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                u: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        x:           [N, node_dim]    node features
        edge_index:  [2, E]           edge connectivity
        edge_attr:   [E, edge_dim]    edge attributes
        u:           [B, global_dim]  global per‐graph features
        batch:       [N]              which graph each node belongs to
        """
        x2 = self.gat(x, edge_index, edge_attr)  # [N, node_dim]
        x  = self.norm1(x + self.drop(x2))

        # cond = u[batch]
        # x    = self.film(x, cond)
        d    = u[batch, -1:].contiguous()  # [N, 1], scalar dilution per node
        x    = self.film(x, d)

        x2 = self.mlp_out(x)                     # [N, node_dim]
        x  = self.norm2(x + self.drop(x2))
        return x


####### GlobalPNAModel: pool x, ignore dilution completely #######
class GlobalPNAModel(nn.Module):
    def __init__(self, node_dim, g_dim, ec50_dim, hidden_dim=50, num_layers=2, dropout=0.0):
        super().__init__()
        self.ec50_dim = ec50_dim
        self.tail_dim = ec50_dim + 1          # EC50 + dilution
        self.nonbio_dim = g_dim
        self.pool = MultiAggregation(["mean", "std", "max", "min"])
        self.global_mlp = self._build_mlp(
            input_dim   = 4 * node_dim,
            hidden_dim  = hidden_dim,
            output_dim  = g_dim,
            n_layers    = num_layers,
            dropout     = dropout)
        
    def _build_mlp(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SELU())
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
       

    def forward(self, x, edge_index, edge_attr, u, batch):
        aggr   = self.pool(x, batch)                  # [B, 4*node_dim]
        head   = self.global_mlp(aggr)                # [B, g_dim]
        tail   = u[:, -self.tail_dim:]                # [B, ec50_dim+1]  (frozen)
        return torch.cat([head, tail], dim=1)         # [B, g_dim+ec50_dim+1]

ec50_dim   = 32              # whatever your vector length is
g_dim      = 201               # example size of trainable global part
global_dim = g_dim + ec50_dim + 1

def get_graphnet_layer(node_dim, edge_dim, global_dim,
                       hidden_dim=50, dropout=0.0) -> MetaLayer:
    edge_net   = EdgeFiLMModel(edge_dim, node_dim, global_dim, hidden=hidden_dim, n_layers=2, dropout=dropout)
    node_net   = NodeAttnFiLM(node_dim, global_dim, n_heads=5, hidden=hidden_dim, dropout=dropout)
    global_net = GlobalPNAModel(
        node_dim   = node_dim,
        g_dim      = g_dim,
        ec50_dim   = ec50_dim,
        hidden_dim = hidden_dim,
        num_layers = 2,
        dropout    = dropout,
    )

    return MetaLayer(edge_net, node_net, global_net)

####### GraphNets: stack MetaLayer(s) and forward to u #######

class GraphNets(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 global_dim: int,
                 hidden_dim: Optional[int] = 50,
                 depth: Optional[int] = 3,
                 dropout: Optional[float] = 0.0,
                 **kwargs):
        super(GraphNets, self).__init__()
        self.node_dim   = node_dim
        self.edge_dim   = edge_dim
        self.global_dim = global_dim
        self.depth      = depth
        self.dropout    = dropout

        self.layers = nn.ModuleList([
            get_graphnet_layer(node_dim   = node_dim,
                               edge_dim   = edge_dim,
                               global_dim = global_dim,
                               hidden_dim = hidden_dim,
                               dropout    = dropout)
            for _ in range(depth)
        ])

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        x, edge_index, edge_attr, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            data.batch,
        )
        # At each layer, edge_net and node_net see only the *scalar* dilution (last coord of u),
        # while global_net rebuilds u from pooled x alone.
        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)
        return u