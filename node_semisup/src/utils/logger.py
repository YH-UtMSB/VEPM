import os


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

