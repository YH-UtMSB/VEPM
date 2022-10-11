import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN_layer(nn.Module):
    def __init__(self, ft_in: int, ft_ot: int, dropout=0., act=F.relu, bias=True):
        """
        Inputs:
            ft_in: [int] dimensionality of input features
            ft_ot: [int] dimensionality of output features
        """
        super(GCN_layer, self).__init__()
        
        # hyperparameters
        self.ft_in = ft_in
        self.ft_ot = ft_ot
        self.dropout = dropout
        self.act = act
        self._bias = bias
        # learnable
        self.linear = nn.Linear(self.ft_in, self.ft_ot, bias=bias)

        self._reset_parameters()
    

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self._bias:
            stdev = 1. / math.sqrt(self.linear.weight.size(1))
            self.linear.bias.data.uniform_(-stdev, stdev)

    
    def forward(self, adj, node_fts):
        # dropout
        h = F.dropout(node_fts, self.dropout, self.training)
        # aggregation
        h = torch.sparse.mm(adj, h)
        # transformation - nonlinearity
        out = self.act(self.linear(h))

        return out






