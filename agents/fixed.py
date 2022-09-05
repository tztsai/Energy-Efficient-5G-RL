from gym.spaces import MultiDiscrete


class FixedPolicy:
    fixed_action = [
        2,  # does not switch on/off antennas
        0,  # does not sleep
        2   # accept new connections
    ]
    
    def __init__(self, action_dims, num_agents):
        self.act_space = MultiDiscrete(action_dims)
        self.num_agents = num_agents
        assert self.fixed_action in self.act_space

    def act(self, *_, **__):
        return [self.fixed_action for _ in range(self.num_agents)]
        # return np.array([np.random.randint(low, high + 1) for low, high in
        #                  zip(self.act_space.low, self.act_space.high)])
