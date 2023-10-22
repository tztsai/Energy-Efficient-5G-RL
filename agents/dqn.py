import numpy as np
import torch
from torch import nn
from .mappo.nn.distributions import Categorical


class QNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=128):
        super().__init__()
        
        self.multi_discrete = False
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            if hasattr(action_space, "nvec"):
                self.action_dims = action_space.nvec
            else:
                self.action_dims = action_space.high - action_space.low + 1
            action_dim = np.prod(self.action_dims)
        else:  # discrete + continous
            raise NotImplementedError("Only discrete action spaces are supported.")

        self.net = nn.Sequential(
            nn.Linear(np.array(obs_space.shape).prod(), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        """
        Compute actions from given input.
        :param x: (torch.Tensor) input to network.
        :return actions: (torch.Tensor) actions to take.
        """
        q_values = self.net(x)
        actions = q_values.argmax(dim=-1)
        if self.multi_discrete:
            action_list = []
            for action_dim in self.action_dims:
                action_list.append(actions % action_dim)
                actions = actions // action_dim
            actions = torch.stack(action_list, -1)
        return actions

    def gather(self, x, actions):
        """
        Gather Q-values for given input and actions.
        :param x: (torch.Tensor) input to network.
        :param actions: (torch.Tensor) actions to take.
        :return q_values: (torch.Tensor) Q-values for given input and actions.
        """
        q_values = self.net(x)
        if self.multi_discrete:
            a = 0
            for i, _ in enumerate(self.action_dims):
                a += actions[..., i] * np.prod(self.action_dims[:i])
            return q_values.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        else:
            return q_values.gather(-1, actions)