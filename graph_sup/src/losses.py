import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops

from .utils import log_max

class UnsupGAE(object):
    '''
        computed the unsupervised loss (graph reconstruction + kl) for the batch.
    '''
    def __init__(self, dataparallel=False):
        self.dp = dataparallel
        self.alpha = torch.tensor([1.])
        self.beta = torch.tensor([1.])
    
    def _align_device_with(self, align_tensor):
        setattr(self, 'device', align_tensor.device)
        self.alpha = self.alpha.to(self.device)
        self.beta = self.beta.to(self.device)

    def GraphRecon(self, s, data):
        '''
            compute graph reconstruction loss for ONE graph.
            args:
                s: node-community affiliation.
                data: a graph data.
        '''
        def _BerPo(s):
            return 1 - torch.exp(- torch.mm(s, s.t()))
        
        def _get_pw(data):
            N = data.num_nodes
            Npos = data.edge_index.size(1)
            Nneg = N * (N - 1) - Npos      

            # pos_w: augment positive observations (edges) to the same amount of negative observations (non-edges)
            # if Npos = 0 (scatters) or Nneg = 0 (complete graph), set pos_wt = 1,
            # otherwise, it should be (Nneg / Npos).
            pos_w = float(Nneg / Npos) if (Nneg * Npos) > 0 else 1.

            return pos_w
        
        preds = _BerPo(s)
        pos_w = _get_pw(data)

        # remove self-loops from the edge_index
        edge_index, _ = remove_self_loops(data.edge_index)

        # create the graph label (a binary adjacency matrix)
        adj_label = torch.sparse_coo_tensor(
            edge_index, torch.ones((edge_index.size(1),)),
            torch.Size([data.num_nodes, data.num_nodes]), device=self.device
        ).to_dense()

        # compute weighted nll
        pos_labels = pos_w * adj_label
        neg_labels = 1. - adj_label - torch.eye(data.num_nodes, device=self.device)

        LL = pos_labels * log_max(preds) + neg_labels * log_max(1. - preds)
        return - LL.sum() / (pos_labels + neg_labels).sum()
    
    def KLD(self, lbd, kappa):
        '''
            compute the KL divergence on one graph.
        '''
        eulergamma = 0.5772
        N = lbd.size(0)

        KL_Part1 = eulergamma * (1 - kappa.pow(-1)) + log_max(lbd / kappa) + 1 + self.alpha * torch.log(self.beta)
        KL_Part2 = - torch.lgamma(self.alpha) + (self.alpha - 1) * (log_max(lbd) - eulergamma * kappa.pow(-1))
        KL_Part3 = - self.beta * lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))

        nKL = KL_Part1 + KL_Part2 + KL_Part3 
        return - nKL.mean() / N
    
    def __call__(self, z, lbd, kappa, data_batch):
        # "de-batch" the data batch
        if self.dp:
            data_list = data_batch
        else:
            data_list = data_batch.to_data_list()
        
        self._align_device_with(z)

        # de-batch z, lbd and kappa
        chunk_sizes = list(map(lambda data: data.num_nodes, data_list))
        z_list = z.split(chunk_sizes, dim=0)
        lbd_list = lbd.split(chunk_sizes, dim=0)
        kappa_list = kappa.split(chunk_sizes, dim=0)

        # compute graph reconstruction loss for the batch
        loss_gre = [self.GraphRecon(z, data) for (z, data) in zip(z_list, data_list)]
        loss_gre = torch.stack(loss_gre).mean()

        # compute kl divergence for the batch
        loss_kld = [self.KLD(lbd, kappa) for (lbd, kappa) in zip(lbd_list, kappa_list)]
        loss_kld = torch.stack(loss_kld).mean()

        return loss_gre + 0.01 * loss_kld