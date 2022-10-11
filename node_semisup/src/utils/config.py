import argparse
import torch

def arg_parse():
    parser = argparse.ArgumentParser(description='VEPM for WikiCS, hyperparams.')

    # global hyperparams
    parser.add_argument('--seed', type=int, default=1994)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--Epch-pretrain', type=int, default=1500, help='unsuper pretrain epochs.')
    parser.add_argument('--Epch-loadmodel', type=int, default=1500)
    parser.add_argument('--Epch-finetune', type=int, default=200,  help='supervised-train epochs.')
    parser.add_argument('--Num-communities', type=int, default=8)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--split-id', type=int, default=0, help='legitimate range is from 0 to 19.')
    parser.add_argument('--fast-mode', action='store_true', default=False, help='inference without setting models to eval()')

    # dirs
    parser.add_argument('--logdir', type=str, default='./test_logs')
    parser.add_argument('--modeldir', type=str, default='./saved_models')
    parser.add_argument('--datapath', type=str, default='/data/datasets/GraphDatasets/WikiCS')
    # parser.add_argument('--datapath', type=str, default='/work2/06083/ylhe/Data/WikiCS')
    
    # INFnet parameters
    parser.add_argument('--INFnet-lr', type=float, default=1e-3)
    parser.add_argument('--INFnet-dp', type=float, default=0., help='COMEncoder dropout rate.')
    parser.add_argument('--INFnet-l2', type=float, default=0., help='weight decay for Adam.')

    # GENnet parameters
    parser.add_argument('--GENnet-lr', type=float, default=1e-2)
    parser.add_argument('--GENnet-dp', type=float, default=0.5)
    parser.add_argument('--GENnet-l2', type=float, default=5e-4)
    parser.add_argument('--GENnet-tau', type=float, default=1., help='community softmax temperature.')
    parser.add_argument('--GENnet-hid', type=int, default=64, help='overall hidden dimension.')
    parser.add_argument('--BankLayers', type=int, default=2, help='the total hidden layers for ComGNNBank.')
    parser.add_argument('--xCat-param', type=float, default=0.03, help='x, phi concatenation parameter.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda:' + str(args.gpu_id) if args.cuda else 'cpu'

    return args
