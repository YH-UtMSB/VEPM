from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import GCN_layer
from ..utils import log_max, sector


###################################### Community Encoder ######################################

class COMEncoder(nn.Module):
    def __init__(self, dims, dropout):
        """
        Inputs:
            dims: [list] dimensionality of each layer, include input layer
            dropout: [float] dropout rate
        """
        super(COMEncoder, self).__init__()
        self.dims = dims
        self.n_layers = len(dims) - 1
        
        # define layers
        self.dropout = dropout
        self.act = nn.Softplus()
        self.bias = False
        self.GCLayers = nn.ModuleList([GCN_layer(dims[0], dims[1], 0., self.act, self.bias)])
        for i in range(1, self.n_layers):
            ft_in = dims[i]
            ft_ot = dims[i+1] if (i+1) < self.n_layers else (dims[i+1] + 1)
            self.GCLayers.append(GCN_layer(ft_in, ft_ot, self.dropout, self.act, self.bias))

        self.deterministic_encoder = True
    
    def reparameterize(self, lbd, kappa):
        random = not self.deterministic_encoder
        if random and self.training:
            # weibull reparameterization: phi = lbd * (- ln(1 - u)) ^ (1/k), u ~ uniform(0,1)
            u = torch.rand_like(lbd)
            phi = lbd * (- log_max(1 - u)).pow(1 / kappa)
        else:
            phi = lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))

        return phi
    
    def forward(self, adj, x):
        # amortized learning
        h = x
        for i in range(len(self.GCLayers)):
            h = self.GCLayers[i](adj, h)
        lbd, kappa = h.split([self.dims[-1], 1], dim=-1)
    
        # reparameterize
        phi = self.reparameterize(lbd, kappa+0.1)
        
        return phi, lbd, kappa+0.1


###################################### Edge Partitioner ######################################

class EdgePart(nn.Module):
    def __init__(self, K, dropout=0., **kwargs):
        super(EdgePart, self).__init__()
        # define number of communities
        self.K = K
        self.dropout = dropout
        self.tau = kwargs['tau'] if 'tau' in kwargs else 1.
    
    def epart(self, phi, adj):
        """ generate (sparse, differentiable) community-specific social adjacency matrices """
        indices = adj._indices()

        def _edge_weight(phi, ind):
            row_ind, col_ind = ind
            phi_row, phi_col = phi[row_ind].unsqueeze(1), phi[col_ind].unsqueeze(2)
            return torch.bmm(phi_row, phi_col).flatten()
        
        def edge_weight(phi, indices):
            """ due to memory constraint, not to compute edge weights all at once """
            max_pairs = 5000
            ind_ = torch.split(indices, max_pairs, dim=1)
            return torch.cat([_edge_weight(phi, ind) for ind in ind_])

        # compute community graph factors      
        phi_ = sector(phi, self.K)
        eg_counts  = torch.stack([edge_weight(phi, indices) for phi in phi_], dim=0)
        eg_weight  = F.softmax(eg_counts / self.tau, dim=0)
        eg_weight_ = torch.unbind(eg_weight, dim=0)

        adj_K_ = list(map(
            lambda x: torch.sparse_coo_tensor(adj._indices(), x, adj.shape, requires_grad=True), eg_weight_
        ))
        # row-normalize sparse community factor graph adjacency mats
        tau_row = 0.5
        adj_K_ = [torch.sparse.softmax(adj_k / tau_row, dim=1) for adj_k in adj_K_]

        return adj_K_
 
        
    def forward(self, phi, adj):
        phi = F.dropout(phi, self.dropout, self.training)
        return self.epart(phi, adj)


###################################### Community GNN bank ######################################
class ComGNN(nn.Module):
    def __init__(self, ft_in, ft_hid, n_layers, dropout, act) -> None:
        super(ComGNN, self).__init__()
        self.layers = nn.ModuleList([GCN_layer(ft_in, ft_hid, dropout, act)])
        for i in range(1, n_layers):
            self.layers.append(GCN_layer(ft_hid, ft_hid, dropout, act))
    
    def forward(self, adj, x):
        h = x
        for m in self.layers:
            h = m(adj, h)
        return h


class ComGNNBank(nn.Module):
    """ a K-head GCN, each one learns from a graph factor A_k """
    def __init__(self, ft_in, ft_hid_, n_layers, **kwargs):
        super(ComGNNBank, self).__init__()
        self.layers = nn.ModuleList(
            list(map(lambda ft_hid: ComGNN(ft_in, ft_hid, n_layers, **kwargs), ft_hid_))
        )
    
    def forward(self, x, adj_K_):
        assert len(adj_K_)== len(self.layers), '#adj neq to #head.'
        out = [m(adj_k, x) for (m, adj_k) in zip(self.layers, adj_K_)]
        return torch.cat(out, dim=-1)


###################################### Representation Composer ######################################

class RepComp(nn.Module):
    """ a single-layer GCN """
    def __init__(self, ft_in, ft_out, **kwargs):
        super(RepComp, self).__init__()
        self.m = GCN_layer(ft_in, ft_out, act=lambda x: x, **kwargs)
    
    def forward(self, adj, x):
        out = self.m(adj, x)
        return F.log_softmax(out, dim=1)