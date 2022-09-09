import copy
import atexit
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    return torch.from_numpy(np.asarray(input))

class RollingStats:
    def __init__(self, name):
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
        