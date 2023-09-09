import numpy as np
from torch import nn


class QNetwork(nn.Sequential):
    def __init__(self, env):
        super().__init__(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )
