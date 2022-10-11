import torch
import torch.nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp

from ..utils import log_max


class LossUnsuper(object):
    def __init__(self, device, **kwargs) -> None:
        super(LossUnsuper, self).__init__()
        self.device = device
        self.alpha = kwargs['gamma_prior_shape'] if 'gamma_prior_shape' in kwargs else 1.
        self.alpha = torch.tensor([self.alpha]).to(self.device)
        self.beta = kwargs['gamma_prior_rate'] if 'gamma_prior_rate' in kwargs else 1.
        self.beta = torch.tensor([self.beta]).to(self.device)
    
    def L_egen(self, phi, gt_data):

        def _get_sliced_pred(phi, selected_rows):
            """ reconstruct several rows of the adjacency matrix """
            return 1 - torch.exp( - torch.mm(phi[selected_rows], phi.t()))

        def _balanced_BCE(preds, labels, pos_wt):
            loss = - (pos_wt * labels * log_max(preds) + (1 - labels) * log_max(1 - preds))
            return loss.sum()

        def _sliced_bBCE(phi, gt_data, Nchunks=16):
            # prepare the row indices for slicing
            selected_rows_ = np.array_split(np.arange(gt_data.Nnodes), Nchunks)
            
            # generate the label matrix
            row, col = gt_data.data.edge_index.cpu().numpy()
            values = np.ones_like(row)
            adj_label = sp.csr_matrix((values, (row, col)), shape=(gt_data.Nnodes, gt_data.Nnodes))

            loss = 0. 
            for selected_rows in selected_rows_:
                labels = adj_label[selected_rows, :].toarray()
                labels = torch.from_numpy(labels).to(self.device)
                preds  = _get_sliced_pred(phi, selected_rows)
                loss += _balanced_BCE(preds, labels, gt_data.pos_weight)

            return loss
        
        loss_egen = _sliced_bBCE(phi, gt_data) / gt_data.Legen_divisor

        return loss_egen
    
    def L_kl(self, kappa, lbd):
        eulergamma = 0.5772

        KL_Part1 = eulergamma * (1 - kappa.pow(-1)) + log_max(lbd / kappa) + 1 + self.alpha * torch.log(self.beta)
        KL_Part2 = - torch.lgamma(self.alpha) + (self.alpha - 1) * (log_max(lbd) - eulergamma * kappa.pow(-1))
        KL_Part3 = - self.beta * lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))

        negKL = KL_Part1 + KL_Part2 + KL_Part3 
        return -negKL.mean()

    def __call__(self, phi, lbd, kappa, gt_data):
        return self.L_egen(phi, gt_data) + 0.0 * self.L_kl(kappa, lbd) / gt_data.Nnodes




