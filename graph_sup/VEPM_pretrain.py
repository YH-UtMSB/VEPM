import os
from easydict import EasyDict as edict
import numpy as np
from glob import glob
from tqdm import tqdm

# pytorch
import torch
from torch.optim import Adam
import torch.nn.functional as F

# pyg
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric import seed_everything
from torch_geometric.utils import degree
from torch_geometric.nn import DataParallel

# vepm
from src.nn import InfNet
from src.losses import UnsupGAE
from src.config import config


def pretrain(loader, infnet, optim, loss, device=None):
    infnet.train()
    epoch_losses = []

    for data_batch in loader:
        if not isinstance(data_batch, list):
            data_batch = data_batch.to(device)

        # process node features
        if not isinstance(data_batch, list):
            if data_batch.x is None:
                data_batch.x = torch.ones((data_batch.num_nodes, 1), dtype=torch.float32, device=device)
        else:
            for data in data_batch:
                if data.x is None:
                    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=data.edge_index.device)

        s, lbd, kappa = infnet(data_batch)

        _loss = loss(s, lbd, kappa, data_batch)
        optim.zero_grad()
        _loss.backward()
        optim.step()

        epoch_losses.append(_loss.item())
    
    return np.mean(epoch_losses)


def main():
    args = config()

    # set random seed
    seed_everything(args.seed)

    # load data, intialize dataloader
    dataset = TUDataset(root=args.datapath, name=args.dataset).shuffle()
    DL = DataListLoader if args.parallel_ptInfNet else DataLoader
    dataloader = DL(dataset, batch_size=args.batch_size)

    # initialize model, optimizer and loss
    in_dim = max(dataset.num_features, 1)
    kwargs = edict()
    kwargs.model = {'in_dim': in_dim, 'hid_dims': [32, 16], 'dropout': args.InfNet_dropout}
    kwargs.optim = {'lr': args.InfNet_lr, 'weight_decay': args.InfNet_l2}
    
    infnet = InfNet(**kwargs.model).to(args.device)

    # load the latest pretrained model
    curr_epoch = 0
    models = glob(os.path.join(args.modelpath, f'InfNet_*_{args.dataset}.pt'))
    get_epoch = lambda f: int(f.split('ep')[0].split('_')[-1])
    if models:
        pt_epochs = list(map(get_epoch, models))
        curr_epoch = np.max(pt_epochs)
        saved_path = os.path.join(args.modelpath, f'InfNet_{curr_epoch}ep_{args.dataset}.pt')
        infnet.load_state_dict(torch.load(saved_path, map_location=args.device))
        print('==========')
        print(f'load model from {saved_path}')
        print('==========')
    

    if args.parallel_ptInfNet:
        device_ids = (np.arange(torch.cuda.device_count()) + args.gpu_id) % torch.cuda.device_count()
        infnet = DataParallel(infnet, device_ids=device_ids.tolist(), output_device=args.device)
    
    optim = Adam(infnet.parameters(),**kwargs.optim)
    loss = UnsupGAE(dataparallel=args.parallel_ptInfNet)

    # start pretraining!
    with tqdm(total=(args.pt_epochs - curr_epoch), desc='(T)') as pbar:
        for ep in range(curr_epoch+1, args.pt_epochs+1):
            ls = pretrain(dataloader, infnet, optim, loss, device=args.device)
            pbar.set_postfix_str(f'loss: {ls:.4f}')
            pbar.update()

            if ep % 250 == 0:
                save_name = os.path.join(args.modelpath, f'InfNet_{ep}ep_{args.dataset}.pt')
                if args.parallel_ptInfNet:
                    torch.save(infnet.module.state_dict(), save_name)
                else:
                    torch.save(infnet.state_dict(), save_name)


if __name__ == '__main__':
    main()