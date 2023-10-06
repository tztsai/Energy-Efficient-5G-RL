import numpy as np
from torch import nn


class QNetwork(nn.Sequential):
    def __init__(self, obs_space, action_space):
        super().__init__(
            nn.Linear(np.array(obs_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_space.n),
        )
