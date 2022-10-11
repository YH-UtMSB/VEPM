from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import COMEncoder, EdgePart, ComGNNBank, RepComp


###################################### Inference Net ######################################
class INFnet(nn.Module):
    def __init__(self, dims, dropout):
        super(INFnet, self).__init__()
        self.COMEncoder = COMEncoder(dims, dropout)
    
    def forward(self, adj, x):
        phi, lbd, kappa = self.COMEncoder(adj, x)
        return phi, lbd, kappa


################################# Generative-Predictive Net ##################################
class GENnet(nn.Module):
    def __init__(self, K, ft_in, ft_hid_overall, n_classes, comgnn_nlayers, **kwargs):
        super(GENnet, self).__init__()

        # get additional params
        self.dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.5
        self.tau = kwargs['tau'] if 'tau' in kwargs else 1.
        self.act = kwargs['act'] if 'act' in kwargs else F.relu

        # compute the hid dim for each ComGNN
        rr = ft_hid_overall % K
        self.ft_hid_ = np.ones(K) * (ft_hid_overall // K)
        self.ft_hid_ += np.append(np.ones(rr), np.zeros(K - rr))
        self.ft_hid_ = self.ft_hid_.astype(np.int64).tolist()

        # forward propagation
        self.EdgePart = EdgePart(K, tau=self.tau)
        self.ComGNNBank = ComGNNBank(ft_in, self.ft_hid_, comgnn_nlayers, dropout=self.dropout, act=self.act)
        self.RepComp = RepComp(ft_hid_overall, n_classes, dropout=self.dropout)

    def forward(self, phi, adj, xhat):
        """
            (phi, adj) --> {adj_k}
            ({adj_k}, xhat) --> H
            (adj, H) --> yhat
        """
        adj_K_ = self.EdgePart(phi, adj)
        H = self.ComGNNBank(xhat, adj_K_)
        yhat = self.RepComp(adj, H)
        return yhat



