from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

import numpy as np
from easydict import EasyDict as edict

from .layers import GINConv


class PredNet(nn.Module):
    def __init__(self, N_coms, in_dim, emb_dim, N_classes, pooltype='sum', n2g_coverage='half', **kwargs):
        super(PredNet, self).__init__()
        self.kwargs = kwargs
        self.N_coms = N_coms
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.N_classes = N_classes
        self.n2g_coverage = n2g_coverage

        # modules to get node representations
        self.EdgePart = EdgePart(**self.configs.EdgePart)
        self.ComGNNBank = ComGNNBank(**self.configs.ComGNNBank)
        self.RepComposer = RepComposer(**self.configs.RepComposer)

        # graphpool determination
        if pooltype == 'sum':
            self.graphpool = global_add_pool
        elif pooltype == 'mean':
            self.graphpool = global_mean_pool
        else:
            raise ValueError(f'pooling mthd {pooltype} is not implemented.')

        # node to graph (n2g) summarization
        self.N_layers = 1 + self.configs.ComGNNBank.N_layers + self.configs.RepComposer.N_layers
        self.predictors = nn.ModuleList()
        for i in range(self.N_layers):
            self.predictors.append(create_predictor(emb_dim, N_classes, self.configs.n2g.final_dropout))
    
    def forward(self, z, data):
        '''
        args:
            z: node-community affiliation, obtained from the InfNet
            data: a minibatch of graph data
        '''
        # network forwarding
        x, edge_index = data.x, data.edge_index

        # step 1: get partitioned edge weights for each community
        edge_weight_list = self.EdgePart(z, edge_index)

        # step 2: collect node representations from ComGNNBank (input + each layer in ComGNNBank)
        hs = self.ComGNNBank(x, edge_index, edge_weight_list)
        hh = hs[-1]

        # step 3: append aggregated node reps from RepComposer to node reps obtained from step 2
        hs += self.RepComposer(hh, edge_index)

        # step 4: graphpooling, applied to all elements in hs
        hgs = [self.graphpool(h, data.batch) for h in hs]

        # step 5: select some of the pooled reps (according to n2g_coverage)
        # and transform them into softmax logits
        hgs = [pred(hg) for (pred, hg) in zip(self.predictors, hgs)]
        if self.configs.n2g.coverage == 'full':
            pass
        elif self.configs.n2g.coverage == 'half':
            # output of ComGNNBank + all layers of RepComposer
            hgs = hgs[-(self.configs.RepComposer.N_layers + 1):]
        elif self.configs.n2g.coverage == 'last':
            # output of RepComposer
            hgs = hgs[-1:]
        else:
            raise ValueError(f'`n2g_coverage` does not support `{self.configs.n2g.coverage}`.')

        # step 6: make final prediction
        logits = 0
        if len(hgs) > 1:
            for hg in hgs:
                logits += F.dropout(hg, self.configs.n2g.final_dropout, training=self.training)
        else:
            logits = hgs[0]

        return F.log_softmax(logits, dim=-1)  

    @property
    def configs(self):
        _configs = edict()

        # EdgePart Configurations
        _configs.EdgePart = edict()
        _configs.EdgePart.N_coms = self.N_coms
        _configs.EdgePart.dropout = self._set_default('ep_dropout', .5)
        _configs.EdgePart.tau = self._set_default('ep_tau', 1.)

        # ComGNNBank Configurations
        _configs.ComGNNBank = edict()
        _configs.ComGNNBank.N_coms = self.N_coms
        _configs.ComGNNBank.in_dim = self.in_dim
        _configs.ComGNNBank.emb_dim = self.emb_dim
        _configs.ComGNNBank.N_layers = self._set_default('N_layers_h', 2)
        _configs.ComGNNBank.dropout = self._set_default('dropout', .5)

        # RepComposer Configurations
        _configs.RepComposer = edict()
        _configs.RepComposer.emb_dim = self.emb_dim
        _configs.RepComposer.N_layers = self._set_default('N_layers_t', 1)
        _configs.RepComposer.dropout = self._set_default('rc_dropout', .5)

        # n2g configuration
        _configs.n2g = edict()
        _configs.n2g.coverage = self.n2g_coverage
        _configs.n2g.final_dropout = self._set_default('final_dropout', .5)

        return _configs

    def _set_default(self, key, default):
        if key in self.kwargs:
            return self.kwargs[key]
        else:
            return default


def create_predictor(emb_dim, N_classes, dropout=0.):
    return nn.Sequential(
        nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb_dim, N_classes)
    )
        

###################################### Edge Partitioner ######################################

class EdgePart(nn.Module):
    def __init__(self, N_coms, dropout, tau=1., max_edges=10000):
        '''
            Partition the edges according to node-community affiliation (s).
            args:
                N_coms: number of communities.
                dropout: dropout on (s) before computing the partition weights.
                tau: softmax temperature along the community dimension.
                max_edges: the edge `chunk` size. (Splitting edges into chunks to prevent OOM.)
        '''
        super(EdgePart, self).__init__()
        self.N_coms = N_coms
        self.dropout = dropout
        self.tau = tau
        self.max_edges = max_edges

    def chunk_computer(self, z, edge_index):
        row, col = edge_index
        z_start, z_end = z[row], z[col]
        return torch.sum(z_start * z_end, dim=1)

    def edge_weigher(self, z, edge_index):
        edge_index_chunks = edge_index.split(self.max_edges, dim=-1)
        return torch.cat([self.chunk_computer(z, indices) for indices in edge_index_chunks])

    def forward(self, z, edge_index):
        '''
            Compute the partition weights for all edges.
            Args:
                z: node-community affiliation matrix (denoted as Z in paper).
                edge_index: sparse adj edge indices.
            Output: a list of edge weights, with each element corresponding to weights in one community graph.
        '''
        z = F.dropout(z, self.dropout, training=self.training)
        z_chunks = z.tensor_split(self.N_coms, dim=-1)

        edge_weight_unnorm = torch.stack([self.edge_weigher(z_k, edge_index) for z_k in z_chunks])        # shape = [N_coms, N_edges]
        edge_weight_list = torch.unbind(F.softmax(edge_weight_unnorm / self.tau, dim=0), dim=0)
        return edge_weight_list


###################################### Multi-community GNN ######################################
class ComGNNBank(nn.Module):
    def __init__(self, N_coms, in_dim, emb_dim, N_layers, dropout, train_eps=False):
        super(ComGNNBank, self).__init__()
        self.train_eps = train_eps
        self.N_coms = N_coms

        # compute emb_dim for each ComGNN
        chunks = np.array_split(np.ones(emb_dim), N_coms)
        com_emb_dims = list(map(lambda x: int(x.sum()), chunks))

        self.ComGNNs = nn.ModuleList([ComGNN(in_dim, ED, N_layers, dropout, self.train_eps) for ED in com_emb_dims])
        self.input_encoder = nn.Linear(in_dim, emb_dim)

    def forward(self, x, edge_index, edge_weight_list):
        # get node reps by community
        outs = []
        for k in range(self.N_coms):
            outs.append(self.ComGNNs[k](x, edge_index, edge_weight_list[k]))
        
        # re-arrange node reps, gather them by the output layer
        # concatenate the outputs from the same layer (<CAVEAT> not including the input layer)
        outs = list(zip(*outs))
        outs = list(map(lambda tup: torch.cat(tup, dim=-1), outs))

        return [self.input_encoder(x)] + outs


def make_gin_conv(Fi_dim, Fo_dim, train_eps):
    return GINConv(nn.Sequential(nn.Linear(Fi_dim, Fo_dim), nn.ReLU(), nn.Linear(Fo_dim, Fo_dim)), train_eps=train_eps)


class ComGNN(nn.Module):
    def __init__(self, in_dim, emb_dim, N_layers, dropout, train_eps=False):
        super(ComGNN, self).__init__()
        self.N_layers  = N_layers
        self.dropout   = dropout

        self.gconvs = nn.ModuleList([make_gin_conv(in_dim, emb_dim, train_eps)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim)])
        for i in range(N_layers - 1):
            self.gconvs.append(make_gin_conv(emb_dim, emb_dim, train_eps))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
    def forward(self, x, edge_index, edge_weight=None):
        hs = [x]
        for i in range(self.N_layers):
            h = self.gconvs[i](hs[i], edge_index, edge_weight=edge_weight)
            h = F.relu(self.batch_norms[i](h))
            hs.append(F.dropout(h, self.dropout, training=self.training))
        return hs[1:]


#################################### Representation Composer ####################################

class RepComposer(nn.Module):
    def __init__(self, emb_dim, N_layers, dropout, train_eps=False):
        super(RepComposer, self).__init__()
        self.N_layers = N_layers
        self.dropout  = dropout

        self.gconvs = nn.ModuleList([make_gin_conv(emb_dim, emb_dim, train_eps)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim)])
        for i in range(N_layers - 1):
            self.gconvs.append(make_gin_conv(emb_dim, emb_dim, train_eps))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
    
    def forward(self, h, edge_index):
        hs = [h]
        for i in range(self.N_layers):
            h = self.gconvs[i](hs[i], edge_index)
            h = F.relu(self.batch_norms[i](h))
            hs.append(F.dropout(h, self.dropout, training=self.training))
        return hs[1:]
        