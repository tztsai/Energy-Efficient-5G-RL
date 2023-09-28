import math
import torch
import numpy as np

from .valuenorm import ValueNorm
from .shared_buffer import SharedReplayBuffer
from .separated_buffer import SeparatedReplayBuffer


def check(input):
    if type(input) is torch.Tensor:
        return input
    return torch.from_numpy(np.asarray(input, dtype=np.float32))

def th2np(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    return a*e**2/2 + (1-a)*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2
