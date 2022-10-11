import os
from easydict import EasyDict as edict
import numpy as np
from glob import glob

# pytorch
import torch
from torch.optim import Adam
import torch.nn.functional as F

# pyg
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.utils import degree

# vepm
from src.nn import InfNet, PredNet
from src.losses import UnsupGAE
from src.config import config
from src.utils import Logger


class Trainer(object):
    def __init__(self, device):
        self.device = device
        self.unsuper_loss = UnsupGAE()
    
    def finetune_prednet(self, loader, models, optims):
        '''
            Keep the infnet fixed, only train the prednet.
        '''
        models.infnet.eval()
        models.prednet.train()

        epoch_losses = []

        for data in loader:
            data = data.to(self.device)

            # process node features
            if data.x is None:
                # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                row, col = data.edge_index
                deg = degree(col, data.num_nodes, dtype=torch.float32)
                data.x = deg.view(data.num_nodes, 1).to(self.device)
            with torch.no_grad():
                z, _, _ = models.infnet(data)
            data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)

            # compute nll loss (classification task)
            yhat = models.prednet(z, data)
            loss = F.nll_loss(yhat.to(torch.float32), data.y)

            # gradient descent
            optims.prednet.zero_grad()
            loss.backward()
            optims.prednet.step()

            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)


    def finetune_infnet(self, loader, models, optims):
        '''
            Keep the prednet fixed, only train the infnet.
        '''
        models.infnet.train()
        models.prednet.eval()

        epoch_losses = []
        
        for data in loader:
            data = data.to(self.device)

            # process node features
            if data.x is None:
                # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                row, col = data.edge_index
                deg = degree(col, data.num_nodes, dtype=torch.float32)
                data.x = deg.view(data.num_nodes, 1).to(self.device)
            z, lbd, kappa = models.infnet(data)
            data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)

            # compute nll loss (classification task)
            yhat = models.prednet(z, data)
            loss_super = 0.1 * F.nll_loss(yhat.to(torch.float32), data.y)

            # compute unsuper loss
            loss_unsup = self.unsuper_loss(z, lbd, kappa, data)

            loss = 0.1 * (loss_super + loss_unsup)

            # gradient descent
            optims.infnet.zero_grad()
            loss.backward()
            optims.infnet.step()

            epoch_losses.append(loss.item())

        return np.mean(epoch_losses)
    

    def finetune_joint(self, loader, models, optims):
        '''
            Joint finetune.
        '''
        models.infnet.train()
        models.prednet.train()

        epoch_losses = []

        for data in loader:
            data = data.to(self.device)

            # process node features
            if data.x is None:
                # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                row, col = data.edge_index
                deg = degree(col, data.num_nodes, dtype=torch.float32)
                data.x = deg.view(data.num_nodes, 1).to(self.device)
            z, lbd, kappa = models.infnet(data)
            data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)

            # compute nll loss (classification task)
            yhat = models.prednet(z, data)
            loss_super = F.nll_loss(yhat.to(torch.float32), data.y)

            # compute unsuper loss
            loss_unsup = self.unsuper_loss(z, lbd, kappa, data)

            loss = loss_super + 0.01 * loss_unsup

            # gradient descent
            optims.infnet.zero_grad(); optims.prednet.zero_grad()
            loss.backward()
            optims.infnet.step(); optims.prednet.step()

            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)

    
    def evaluate(self, loader, models):
        '''
            val fold acc
        '''
        models.infnet.eval()
        models.prednet.eval()

        num_evaluate = 0
        num_corrects = 0

        for data in loader:
            data = data.to(self.device)

            # process node features
            if data.x is None:
                # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                row, col = data.edge_index
                deg = degree(col, data.num_nodes, dtype=torch.float32)
                data.x = deg.view(data.num_nodes, 1).to(self.device)
            
            with torch.no_grad():
                z, _, _ = models.infnet(data)
                data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)
                yhat = models.prednet(z, data)
            
            # make prediction
            y_pred = yhat.max(1).indices
            num_evaluate += len(y_pred)
            num_corrects += (y_pred.eq(data.y)).sum().cpu().item()

        return num_corrects / num_evaluate


def main(args):

    # set random seed
    seed_everything(args.seed)

    # initialize logger
    LOGPATH = os.path.join(args.logpath, args.dataset, f'{args.n2g_coverage}_{args.PredNet_Nlayers_CGBank}_{args.PredNet_Nlayers_REComp}_{args.pooltype}')
    if args.enable_logger:
        logger = Logger(LOGPATH, f'val{args.fold_id}.log')
    else:
        logger = None

    # load dataset, create 9 train folds and 1 val fold
    dataset = TUDataset(args.datapath, name=args.dataset)
    indices = np.random.RandomState(seed=args.seed).permutation(len(dataset))
    tenfold = np.array_split(indices, 10)
    val_indices = tenfold.pop(args.fold_id)
    trn_indices = np.concatenate(tenfold, axis=0)

    loader_trn = DataLoader(dataset[trn_indices], batch_size=args.batch_size)
    loader_val = DataLoader(dataset[val_indices], batch_size=args.batch_size)

    # record the settings
    if logger is not None:
        logger.record_args(args)
    
    # initialize the model, optimizer
    kwargs = edict()
    kwargs.models = edict()
    kwargs.optims = edict()

    models = edict()
    optims = edict()

    print(dataset.num_features)

    # the infnet
    kwargs.models.infnet = {
        'in_dim': max(dataset.num_features, 1),
        'hid_dims': [32, 16], 
        'dropout': args.InfNet_dropout
    }
    kwargs.optims.infnet = {
        'lr': args.InfNet_lr,
        'weight_decay': args.InfNet_l2
    }
    models.infnet = InfNet(**kwargs.models.infnet).to(args.device)
    optims.infnet = Adam(models.infnet.parameters(), **kwargs.optims.infnet)

    # the prednet
    in_dim = max(dataset.num_features, 1) + 16
    kwargs.models.prednet = {
        'N_coms': args.N_coms,
        'in_dim': in_dim,
        'emb_dim': args.PredNet_edim,
        'N_classes': dataset.num_classes,
        'pooltype': args.pooltype,
        'n2g_coverage': args.n2g_coverage,
        'ep_tau': args.EdgePart_tau,
        'N_layers_h': args.PredNet_Nlayers_CGBank,
        'N_layers_t': args.PredNet_Nlayers_REComp
    }
    kwargs.optims.prednet = {
        'lr': args.PredNet_lr,
        'weight_decay': args.PredNet_l2
    }
    models.prednet = PredNet(**kwargs.models.prednet).to(args.device)
    optims.prednet = Adam(models.prednet.parameters(), **kwargs.optims.prednet)

    # load pretrained infnet
    load_path = os.path.join(args.modelpath, f'InfNet_{args.load_epoch}ep_{args.dataset}.pt')
    models.infnet.load_state_dict(torch.load(load_path, map_location=args.device))

    print('==========')
    print(f'load model {load_path}')

    # train the model
    trainer = Trainer(args.device)

    # 1. iterative training

    if args.finetune_scheme == 'Iterative':
        print('==========')
        print('finetune mode: Iterative')
        for i in range(1, args.epochs+1):
            if (i-1) % args.iter_interval == 0:
                loss_inf = trainer.finetune_infnet(loader_trn, models, optims)
                msg1 = f' loss_inf = {loss_inf:.4f},'
            loss_pred = trainer.finetune_prednet(loader_trn, models, optims)
            msg2 = f' loss_pred = {loss_pred:.4f}'

            eval_acc = trainer.evaluate(loader_val, models) * 100
            msg = f'(E) ep {i}, ACC_VAL = {eval_acc:.2f},' + msg1 + msg2
            if logger is not None:
                logger.record(msg)
            else:
                print(msg)
    
    # 2. only train the prednet

    elif args.finetune_scheme == 'PredOnly':
        print('==========')
        print('finetune mode: PredOnly')
        for i in range(1, args.epochs+1):
            loss_pred = trainer.finetune_prednet(loader_trn, models, optims)
            msg1 = f' loss_pred = {loss_pred:.4f}'

            eval_acc = trainer.evaluate(loader_val, models) * 100
            msg = f'(E) ep {i}, ACC_VAL = {eval_acc:.2f},' + msg1
            if logger is not None:
                logger.record(msg)
            else:
                print(msg)
    
    # 3. jointly finetune infnet and prednet

    elif args.finetune_scheme == 'Joint':
        print('==========')
        print('finetune mode: Joint')
        for i in range(1, args.epochs+1):
            loss = trainer.finetune_joint(loader_trn, models, optims)
            msg1 = f' loss_joint = {loss:.4f}'

            eval_acc = trainer.evaluate(loader_val, models) * 100
            msg = f'(E) ep {i}, ACC_VAL = {eval_acc:.2f},' + msg1
            if logger is not None:
                logger.record(msg)
            else:
                print(msg)

    else:
        raise ValueError(f'the finetune scheme `{args.finetune_shceme}` is not identified.')


if __name__ == '__main__':
    args = config()
    main(args)


