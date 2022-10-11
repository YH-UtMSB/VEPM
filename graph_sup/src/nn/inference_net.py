import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ..utils import log_max

class InfNet(nn.Module):
    def __init__(self, in_dim, hid_dims, dropout, gconv_bias=False, **kwargs) -> None:
        super(InfNet, self).__init__()
        self.dropout = dropout
        
        dims = [in_dim] + list(hid_dims)
        # 1 additional dim to store the value of `kappa`, the shape parameter of weibull distribution
        dims[-1] = dims[-1] + 1 

        self.GConvLayers = nn.ModuleList(
            [GCNConv(F_in, F_out, bias=gconv_bias) for (F_in, F_out) in zip(dims[:-1], dims[1:])]
        ) 

        self.random_encoder = kwargs['random_encoder'] if 'random_encoder' in kwargs else False
    

    def reparameterize(self, lbd, kappa):
        '''
            weibull reparameterization: z = lbd * (- ln(1 - u)) ^ (1/kappa), u ~ uniform(0,1)
            z: node-community affiliation.
            lbd: scale parameter, kappa: shape parameter
        '''
        if self.random_encoder and self.training:
            u = torch.rand_like(lbd)
            z = lbd * (- log_max(1 - u)).pow(1 / kappa)
        else:
            z = lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))

        return z

    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # gcn forwarding
        h = F.softplus(self.GConvLayers[0](x, edge_index))
        for gconv in self.GConvLayers[1:]:
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.softplus(gconv(h, edge_index))
        
        # split the output of gcn into lbd and kappa, kappa is a scalar for each node.
        lbd, kappa = h.split([h.size(1)-1, 1], dim=1)
        
        return self.reparameterize(lbd, kappa + 0.1), lbd, kappa + 0.1