"""Implementation based on the template of ALIGNN, Matformer."""

import numpy as np
import torch
import torch.nn.functional as F
from pydantic.typing import Literal
from torch import nn
from zeoformer.models.transformer import ZeoformerConv

import math
from typing import Optional, Tuple, Union

from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

from typing import Optional

from torch_scatter import gather_csr, scatter, segment_csr

from torch_geometric.utils.num_nodes import maybe_num_nodes

import json
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

from pydantic import BaseSettings as PydanticBaseSettings

class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"

class ZeoformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["zeoformer"]
    node_transformer_layers: int = 4
    ppe_layers: int = 3
    cgcnn_features: int = 92
    edge_features: int = 256
    triplet_input_features: int = 40
    node_features: int = 256
    fc_features: int = 256
    output_features: int = 1
    att_node_head: int = 1

    link: Literal["identity", "log", "logit"] = "identity"
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"

class PPE_Conv(nn.Module):
    def __init__(self, node_features):
        super().__init__()
        self.feature = node_features
        self.lin1 = nn.Linear(node_features, node_features * 2)
        self.lin2 = nn.Linear(node_features, node_features)
        self.lin3 = nn.Linear(node_features, node_features)
        self.bn = nn.BatchNorm1d(node_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x, adj):
        adj = adj + x
        adj2 = self.lin1(self.bn(adj))
        x1, x2 = adj2.chunk(chunks=2, dim=1)
        x1 = self.lin2(x1)
        x2 = self.drop(self.act(x2))
        x = self.lin3(x1 * x2) + x
        return x, adj


class RBF(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 4,
        bins: int = 256,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

class Zeoformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, config: ZeoformerConfig = ZeoformerConfig(name="zeoformer")):
        """Set up att modules."""
        super().__init__()
        self.atom_embedding = nn.Sequential(
            nn.Linear(config.cgcnn_features, config.node_features),
            nn.SiLU(),
            nn.Linear(config.node_features, config.node_features)
        )
        self.egde_embedding = nn.Sequential(
            RBF(
                vmin=-6.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.SiLU(),
        )

        self.ppe_embedding = nn.Sequential(
            nn.Linear(config.cgcnn_features, config.node_features)
        )


        self.node_transformer_layers = nn.ModuleList(
            [
                ZeoformerConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.att_node_head, edge_dim=config.edge_features)
                for _ in range(config.node_transformer_layers)
            ]
        )

        self.ppe_layers = nn.ModuleList(
            [
                PPE_Conv(config.node_features)
                for _ in range(config.ppe_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.node_features), nn.SiLU(),
        )

        self.output = nn.Linear(config.node_features, config.output_features)


    def forward(self, data) -> torch.Tensor:


        atom_features = torch.floor(data.x)  # The integer part is the atomic feature
        atom_features = self.atom_embedding(atom_features)
        ppe_feature = torch.frac(data.x) * 1000  # The decimal part is ppe.
        # Round the number
        edge_feat = - 1 / torch.norm(data.edge_attr, dim=1)
        edge_features = self.egde_embedding(edge_feat)
        ppe_feature = self.ppe_embedding(ppe_feature)

        atom_features = self.node_transformer_layers[0](atom_features, data.edge_index, edge_features)

        # atom_features = self.ppe_layers[0](atom_features)
        atom_features, ppe_feature = self.ppe_layers[0](atom_features, ppe_feature)
        atom_features = self.node_transformer_layers[1](atom_features, data.edge_index, edge_features)

        # atom_features = self.pppe_layers[1](atom_features)
        atom_features, ppe_feature = self.ppe_layers[1](atom_features, ppe_feature)
        atom_features = self.node_transformer_layers[2](atom_features, data.edge_index, edge_features)

        atom_features, ppe_feature = self.ppe_layers[2](atom_features, ppe_feature)
        atom_features = self.node_transformer_layers[3](atom_features, data.edge_index, edge_features)

        # crystal-level readout
        features_total = scatter(atom_features, data.batch, dim=0, reduce="mean")

        crystal_features = features_total + self.fc(features_total)

        crystal_output = self.output(crystal_features)

        return torch.squeeze(crystal_output)/data.num_osda


