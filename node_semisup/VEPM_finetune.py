import os, time
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import WikiCS
from easydict import EasyDict as edict

from src.utils import arg_parse, Logger, load_pretrained_models, preprocess_feat
from src.data_process import dataset_wikics
from src.model import INFnet, GENnet, LossUnsuper


def inference(VEPM_Models, DS, args):
    x = DS.data.x
    adj = torch.sparse_coo_tensor(
        DS.data.edge_index, DS.data.edge_weight, (DS.Nnodes, DS.Nnodes)
    ).to(args.device)
    phi, lbd, kappa = VEPM_Models.INFnet(adj, x)

    xhat = preprocess_feat(x, phi, mu=args.xCat_param)
    yhat = VEPM_Models.GENnet(phi, adj, xhat)

    return yhat, phi, lbd, kappa


def warm_up(VEPM_Models, VEPM_Optims, DS, epoch, args):
    t0 = time.time()
    VEPM_Models.INFnet.eval()
    VEPM_Models.GENnet.train()

    yhat, _, _, _ = inference(VEPM_Models, DS, args)
    acc_val, acc_test = evaluation(VEPM_Models, DS, args, yhat=yhat)

    train_mask = DS.data.train_mask[:,args.split_id].flatten()
    train_idx = torch.where(train_mask == 1)[0]
    loss = F.nll_loss(yhat[train_idx], DS.data.y[train_idx])

    if VEPM_Optims.GENnet is not None:
        VEPM_Optims.GENnet.zero_grad()
        loss.backward()         # need to add 'retain_graph = True' for joint training.
        VEPM_Optims.GENnet.step()
    
    t = time.time() - t0
    msg = f'in epoch {epoch}, task_loss = {loss.item():.4f}, acc_val = {acc_val*100:.2f}, acc_test = {acc_test*100:.2f}, time elapsed = {t:.2f}s.'
    args.logger.record(msg)


def finetune(VEPM_Models, VEPM_Optims, DS, VEPM_loss, epoch, args):
    t0 = time.time()
    VEPM_Models.INFnet.train()
    VEPM_Models.GENnet.train()

    yhat, phi, lbd, kappa = inference(VEPM_Models, DS, args)
    acc_val, acc_test = evaluation(VEPM_Models, DS, args, yhat=yhat)

    # accumulating loss
    loss = 0.
    train_mask = DS.data.train_mask[:,args.split_id].flatten()
    train_idx = torch.where(train_mask == 1)[0]
    loss += F.nll_loss(yhat[train_idx], DS.data.y[train_idx])
    loss += 0.01 * VEPM_loss(phi, lbd, kappa, DS)

    for mod in VEPM_Optims:
        VEPM_Optims[mod].zero_grad()
    loss.backward()
    VEPM_Optims.GENnet.step()

    if epoch % 10 == 0:
        VEPM_Optims.INFnet.step()
    
    t = time.time() - t0
    msg = f'in epoch {epoch}, task_loss = {loss.item():.4f}, acc_val = {acc_val*100:.2f}, acc_test = {acc_test*100:.2f}, time elapsed = {t:.2f}s.'
    args.logger.record(msg)
    
    

def evaluation(VEPM_Models, DS, args, **kwargs):
    if not args.fast_mode:
        for mod in VEPM_Models:
            VEPM_Models[mod].eval()
        yhat, _, _, _ = inference(VEPM_Models, DS, args)
    else:
        yhat = kwargs['yhat']
    
    yhat = yhat.max(1)[1].type(DS.data.y.dtype)
    corrects = yhat.eq(DS.data.y).type(torch.double)

    val_mask = DS.data.val_mask[:,args.split_id].flatten()
    test_mask = DS.data.test_mask

    acc_val = (corrects * val_mask).sum() / val_mask.sum()
    acc_test = (corrects * test_mask).sum() / test_mask.sum()

    return acc_val, acc_test
    


if __name__ == '__main__':
    args = arg_parse()
    dargs = vars(args)
    
    # set seed for pytorch, numpy and python
    seed_everything(args.seed)

    # initialize Logger
    logdir = os.path.join(args.logdir, f'pt-{args.Epch_loadmodel}-epch-finetune')
    logger = Logger(logdir=logdir, logfile=f'split-id-{args.split_id}.log')
    args.logger = logger

    # load and preprocess dataset (core data is already moved to cuda)
    raw_ds = WikiCS(root=args.datapath)
    proc = dataset_wikics(device=args.device)
    DS = proc(raw_ds)
    ft_in = DS.data.x.shape[1]
    n_classes = DS.data.y.max().item() + 1

    # initialize model(s) and optimizer(s)
    VEPM_Models = edict(); VEPM_Optims = edict()
    args.INFnet_dims = [ft_in, 32, 16]
    
    VEPM_Models.INFnet = INFnet(
        dims = args.INFnet_dims, 
        dropout = args.INFnet_dp
    ).to(args.device)

    VEPM_Models.GENnet = GENnet(
        K = args.Num_communities,
        ft_in = ft_in + 16,
        ft_hid_overall = args.GENnet_hid,
        n_classes = n_classes,
        comgnn_nlayers = args.BankLayers,
        tau = args.GENnet_tau,
        dropout = args.GENnet_dp
    ).to(args.device)

    for mod in VEPM_Models:
        VEPM_Optims[mod] = torch.optim.Adam(
            VEPM_Models[mod].parameters(),
            lr = dargs[f'{mod}_lr'],
            weight_decay = dargs[f'{mod}_l2']
        )
    
    # start training
    VEPM_loss = LossUnsuper(device=args.device)
    args.logger.record_args(args)
    
    load_pretrained_models(
        Models = VEPM_Models,
        save_path = args.modeldir,
        pretrain_epoch = args.Epch_loadmodel,
        device = args.device
    )

    for epoch in range(1, 51):
        warm_up(VEPM_Models, VEPM_Optims, DS, epoch, args)
    for epoch in range(51, args.Epch_finetune):
        finetune(VEPM_Models, VEPM_Optims, DS, VEPM_loss, epoch, args)
    