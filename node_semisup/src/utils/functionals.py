import torch
import os
import numpy as np
from glob import glob


SMALL = 1e-10
def log_max(input, SMALL=SMALL):
    device = input.device
    input_ = torch.max(input, torch.tensor([SMALL]).to(device))
    return torch.log(input_)


def sector(embeddings, num_split):
    """ partition the dimension of Phi to communities """
    dim = embeddings.shape[-1]
    # create split sections: [len+1, len+1, ..., len+1, len, len, ..., len]
    rr = dim % num_split
    coms_per_meta = np.ones(num_split) * (dim // num_split)
    coms_per_meta += np.append(np.ones(rr), np.zeros(num_split - rr))
    coms_per_meta = coms_per_meta.astype(np.int64).tolist()
    return embeddings.split(coms_per_meta, dim=-1)


def normalize_row(unnorm_mat: torch.Tensor):
    device = unnorm_mat.device
    sparse = unnorm_mat.is_sparse
    if sparse:
        degrees = torch.sparse.sum(unnorm_mat, 1)._values()
    else:
        degrees = unnorm_mat.sum(1)
    dinv = degrees.pow(-1.).flatten()
    dinv[torch.isinf(dinv)] = 0.
    dinv_diag = torch.diag(dinv)
    if sparse:
        norm_mat = torch.sparse.mm(unnorm_mat.t(), dinv_diag).t().to_sparse()
    else:
        norm_mat = torch.mm(dinv_diag, unnorm_mat)

    return norm_mat


def preprocess_feat(feat, phi, mu=3e-4, normalize=False):
    """create feat = cat(phi*mu', feat) for task-learning"""
    mag_div = feat.sum(-1) / phi.sum(-1)
    out_fts = torch.cat([mu * mag_div.unsqueeze(-1) * phi, feat], dim=-1)
    if normalize:
        out_fts = normalize_row(out_fts)
    return out_fts


def load_pretrained_models(Models, save_path, pretrain_epoch, device):
    """
    load the pretrained models (for a specific pretraining epoch),
    """
    saved_model_ls = glob(save_path + os.sep + f'*_{pretrain_epoch}epochs*.pt')

    pretrained_models = {}
    for model in saved_model_ls:
        key = model.split(os.sep)[-1].split('_')[0]
        pretrained_models[key] = model
    
    for key in pretrained_models:
        Models[key].load_state_dict(
            torch.load(pretrained_models[key], map_location=device)
        )
        print(f'{key} successfully loaded')

