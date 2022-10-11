import os, time
from glob import glob
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.datasets import WikiCS
from easydict import EasyDict as edict

from utils import arg_parse, Logger, load_pretrained_models
from data_process import dataset_wikics
from model import INFnet, LossUnsuper


def unsuper_pretrain(VEPM_Models, VEPM_Optims, DS, VEPM_loss, epoch, args):
    t0 = time.time()
    VEPM_Models.INFnet.train()

    x = DS.data.x
    adj = torch.sparse_coo_tensor(
        DS.data.edge_index, DS.data.edge_weight, (DS.Nnodes, DS.Nnodes)
    ).to(args.device)
    phi, lbd, kappa = VEPM_Models.INFnet(adj, x)

    loss = VEPM_loss(phi, lbd, kappa, DS)
    if VEPM_Optims.INFnet is not None:
        VEPM_Optims.INFnet.zero_grad()
        loss.backward()
        VEPM_Optims.INFnet.step()
    
    t = time.time() - t0
    msg = f'in epoch {epoch}, pt-loss = {loss.item():.4f}, time = {t:.2f}s.'
    args.logger.record(msg)



if __name__ == '__main__':
    args = arg_parse()

    # set seed for pytorch, numpy and python
    seed_everything(args.seed)

    # initialize Logger
    logger = Logger(logdir=args.logdir, logfile='pretrain.log')
    args.logger = logger

    # load and preprocess dataset (core data is already moved to cuda)
    raw_ds = WikiCS(root=args.datapath)
    proc = dataset_wikics(device=args.device)
    DS = proc(raw_ds)
    ft_in = DS.data.x.shape[1]

    # initialize model(s) and optimizer(s)
    VEPM_Models = edict(); VEPM_Optims = edict()
    args.INFnet_dims = [ft_in, 32, 16]
    
    VEPM_Models.INFnet = INFnet(
        dims=args.INFnet_dims, 
        dropout=args.INFnet_dp
    ).to(args.device)

    VEPM_Optims.INFnet = torch.optim.Adam(
        VEPM_Models.INFnet.parameters(),
        lr=args.INFnet_lr,
        weight_decay=args.INFnet_l2
    )

    # initialize pretraining loss
    VEPM_loss = LossUnsuper(device=args.device)

    args.logger.record_args(args)

    # start pretraining
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)
    
    # load already pretrained model
    curr_epoch = 0
    pretrained_models = glob(os.sep.join([args.modeldir, '*.pt']))
    if pretrained_models:
        pretrained_epochs = list(map(
            lambda x: int(x.split('epochs')[0].split('_')[-1]), pretrained_models
        ))
        curr_epoch = np.max(pretrained_epochs)
        load_pretrained_models(
            Models=VEPM_Models,
            save_path=args.modeldir,
            pretrain_epoch=curr_epoch,
            device=args.device
        )

    for epoch in range(curr_epoch+1, args.Epch_pretrain+1):
        unsuper_pretrain(VEPM_Models, VEPM_Optims, DS, VEPM_loss, epoch, args)
        for m in VEPM_Models:
            save_name = os.path.join(args.modeldir, f'{m}_unsup_{epoch}epochs.pt')
            torch.save(VEPM_Models[m].state_dict(), save_name)
        if epoch % 250 != 1:
            for m in VEPM_Models:
                os.remove(os.path.join(args.modeldir, f'{m}_unsup_{epoch-1}epochs.pt'))


    


    

