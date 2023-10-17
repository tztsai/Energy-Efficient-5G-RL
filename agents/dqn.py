import numpy as np
from torch import nn
from .mappo.nn.act import ACTLayer


class QNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(np.array(obs_space.shape).prod(), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.act = ACTLayer(action_space, hidden_size)
            
    def forward(self, x):
        h = self.head(x)
        actions, _ = self.act(h, deterministic=True)
        return actions
