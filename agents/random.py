from gym.spaces import MultiDiscrete


class RandomPolicy:
    def __init__(self, action_dims, num_agents):
        self.act_space = MultiDiscrete(action_dims)
        self.num_agents = num_agents

    def act(self, *_, **__):
        return [self.act_space.sample() for _ in range(self.num_agents)]
        # return np.array([np.random.randint(low, high + 1) for low, high in
        #                  zip(self.act_space.low, self.act_space.high)])
