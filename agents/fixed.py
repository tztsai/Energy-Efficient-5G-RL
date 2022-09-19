from gym.spaces import MultiDiscrete

class AlwaysOnPolicy:
    fixed_action = [
        2,  # does not switch on/off antennas
        0,  # does not sleep
        2   # accept new connections
    ]
    
    def __init__(self, action_space, num_agents):
        self.num_agents = num_agents

    def act(self, obs, **__):
        return [self.fixed_action for _ in range(self.num_agents)]
        # return np.array([np.random.randint(low, high + 1) for low, high in
        #                  zip(self.act_space.low, self.act_space.high)])
