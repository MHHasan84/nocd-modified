import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch as th
from torch import nn


from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn import EdgeGATConv
from dgl.nn import GATConv

from nocd.utils import to_sparse_tensor

__all__ = [
    'EGCN',
    'EdgeGATConv',
]


def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)


class EGCN(nn.Module):
    """Graph convolution network.

    References:
        "Semi-superivsed learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """
    def __init__(
        self,
        in_feats,
        edge_feats,
        hidden_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        dropout=0.5, 
        batch_norm=False,
        ):
        super().__init__()
        self.dropout = dropout
        self.edge_gat_conv = EdgeGATConv(in_feats, edge_feats,hidden_feats[0],1,allow_zero_in_degree=True)
        self.gat_conv = GATConv(hidden_feats[0], out_feats,1,allow_zero_in_degree=True)
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_feats
            ]
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)

    def forward(self, graph, feat, edge_feat, get_attention=False):
        if self.dropout != 0:
            feat = sparse_or_dense_dropout(feat, p=self.dropout, training=self.training)
        feat=self.edge_gat_conv(graph,feat,edge_feat,get_attention)
        feat = F.relu(feat)
        if self.batch_norm is not None:
            feat = self.batch_norm[0](feat)
        if self.dropout != 0:
            feat = sparse_or_dense_dropout(feat, p=self.dropout, training=self.training)
        feat=self.gat_conv(graph,feat) 
        return feat

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
