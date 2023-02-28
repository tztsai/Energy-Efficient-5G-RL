import numpy as np
from gym.spaces import MultiDiscrete

class AlwaysOnPolicy:
    def __init__(self, action_space, num_agents, num_ants=64):
        self.num_agents = num_agents
        self.num_ants = num_ants

    def act(self, obs, **__):
        return [[0 if ob[1] > self.num_ants else 1, 0, 2] for ob in obs]
