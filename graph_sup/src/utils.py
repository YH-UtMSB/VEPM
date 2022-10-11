import torch
import torch.nn.functional as F
from glob import glob
import os
import numpy as np


SMALL = 1e-10
NINF = -1e10

def log_max(input, SMALL=SMALL):
    device = input.device
    input_ = torch.max(input, torch.tensor([SMALL]).to(device))
    return torch.log(input_)


def drop_softmax(input, p, dim):
    '''
        first performing dropout on the input, then apply softmax along `dimension`.
    '''
    _device = input.device
    if p < 0 or p >= 1:
        raise ValueError(f'domain `p` out of range: expects to be within [0,1), receives {p}')
    else:
        s = 1. / (1. - p)
    mask_ = torch.from_numpy(np.random.binomial(1, (1. - p), size=input.shape).astype(np.float32)).to(_device)
    # mask_ = torch.from_numpy(np.random.randint(2, size=input.shape).astype(np.float32)).to(_device)
    input = (1. - mask_) * s * input + NINF * mask_
    return F.softmax(input, dim=dim)
            
            
class Logger(object):
    def __init__(self, logdir, logfile):
        super(Logger, self).__init__()
        self.logdir = logdir
        self.logfile = logfile
        if not os.path.exists(logdir):
            os.makedirs(logdir)  
        self.logpath = os.path.join(self.logdir, self.logfile)
    
    def record(self, msg):
        msg = msg + '\n'
        with open(self.logpath, 'a') as f:
            f.write(msg)
        print(msg)
    
    def record_args(self, args):
        for attr, value in sorted(vars(args).items()):
            self.record(f'{attr.upper()}: {value}\n')


