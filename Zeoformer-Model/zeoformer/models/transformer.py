"""Implementation based on the template of Matformer."""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter


class ZeoformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        beta: bool = False,
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(ZeoformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)

        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                            nn.SiLU(),
                                            nn.Linear(out_channels, out_channels))
        self.keyi_update = nn.Sequential(nn.Linear(out_channels * 2, out_channels))
        self.keyj_update = nn.Sequential(nn.Linear(out_channels * 2, out_channels))
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.bn_att = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels)
        self.outatt = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_concate(out)

        return self.softplus(x[1] + self.bn(out))

    def message(self, query_i: Tensor, query_j: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.keyj_update(torch.cat((key_j, edge_attr), dim=-1))
        key_i = self.keyi_update(torch.cat((key_i, edge_attr), dim=-1))
        alpha = (query_i * key_i) / math.sqrt(self.out_channels) + (query_j * key_j) / math.sqrt(self.out_channels)
        out = (self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1)) *
               self.sigmoid(self.layer_norm(self.outatt(torch.cat((value_i, value_j, edge_attr), dim=-1)))))
        out = query_i + out * self.sigmoid(
            self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels))
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

