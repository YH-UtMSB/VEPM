import argparse
import torch
import os

def config():
    parser = argparse.ArgumentParser(description='VEPM hyperparameters.')

    # global parameters
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--fold-id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="MUTAG",
        help='dataset name (default: IMDB-BINARY)')

    # batch (pre)training
    parser.add_argument('--load-epoch', type=int, default=1500)
    parser.add_argument('--parallel-ptInfNet', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    
    # number of communities
    parser.add_argument('--N-coms', type=int, default=4)

    # InfNet parameters
    parser.add_argument('--InfNet-lr', type=float, default=1e-3)
    parser.add_argument('--InfNet-l2', type=float, default=0.)
    parser.add_argument('--InfNet-dropout', type=float, default=0.5)
    
    # PredNet parameters
    parser.add_argument('--PredNet-lr', type=float, default=1e-2)
    parser.add_argument('--PredNet-l2', type=float, default=5e-4)
    parser.add_argument('--PredNet-Nlayers-CGBank', type=int, default=2)
    parser.add_argument('--PredNet-Nlayers-REComp', type=int, default=1)
    parser.add_argument('--PredNet-edim', type=int, default=64)
    parser.add_argument('--PredNet-dropout', type=float, default=0.5)
    parser.add_argument('--EdgePart-dropout', type=float, default=0.)
    parser.add_argument('--pooltype', type=str, default='mean', 
        help='`mean` or `sum`')
    parser.add_argument('--EdgePart-tau', type=float, default=1.)
    parser.add_argument('--n2g-coverage', type=str, default='half',
        help='`full`, `half` or `last`.')

    # some paths
    parser.add_argument('--datapath', type=str, default='/data/datasets/GraphDatasets/TU-pyg',
        help='/work/06083/ylhe/Data/TU-pyg, /data/datasets/GraphDatasets/TU-pyg')
    parser.add_argument('--logpath', type=str, default='./log')
    parser.add_argument('--modelpath', type=str, default='./saved_models')

    # misc
    parser.add_argument('--enable-logger', action='store_true', default=False)
    parser.add_argument('--finetune-scheme', type=str, default='PredOnly',
        help='PredOnly, Joint, Iterative')
    parser.add_argument('--iter-interval', type=int, default=10)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda:' + str(args.gpu_id) if args.cuda else 'cpu'

    if not os.path.exists(args.datapath):
        os.makedirs(args.datapath)

    if not os.path.exists(args.logpath):
        os.makedirs(args.logpath)

    if not os.path.exists(args.modelpath):
        os.makedirs(args.modelpath)

    return args

    
    