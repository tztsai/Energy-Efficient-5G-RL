import copy
import numpy as np
import torch
import torch.nn as nn
from utils import pd, sys, atexit, wraps, defaultdict


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    if type(input) is torch.Tensor:
        return input
    return torch.from_numpy(np.asarray(input, dtype=np.float32))

class StatsCollector:
    def __init__(self, name=None):
        self.name = name
        self.count = 0
        atexit.register(self.save_stats)

    def insert(self, x):
        if self.count == 0:
            self.mean = x
            self.sq_mean = x ** 2
        else:
            self.mean = (self.mean * self.count + x) / (self.count + 1)
            self.sq_mean = (self.sq_mean * self.count + x ** 2) / (self.count + 1)
        self.count += 1

    def get_stats(self):
        return self.mean, np.sqrt(self.sq_mean - self.mean ** 2 + 1e-6)
    
    def save_stats(self):
        mean, std = map(pd.DataFrame, self.get_stats())
        mean.to_csv(f'analysis/{self.name}_mean.csv', index=False)
        std.to_csv(f'analysis/{self.name}_std.csv', index=False)
        

def trace_stats(var):
    def decorator(func):
        scs = defaultdict(StatsCollector)
        code = func.__code__
        def tracer(frame, event, arg):
            if event == 'call' and frame.f_code is code:
                if not torch.is_grad_enabled():
                    sc = scs[frame.f_locals.get('self')]
                    x = frame.f_locals[var].numpy()
                    if sc.name is None:
                        sc.name = var + str(x.shape)
                    sc.insert(x)
        @wraps(func)
        def wrapper(*args, **kwargs):
            _tr = sys.getprofile()
            sys.setprofile(tracer)
            try:
                return func(*args, **kwargs)
            finally:
                sys.setprofile(_tr)
        return wrapper
    return decorator
