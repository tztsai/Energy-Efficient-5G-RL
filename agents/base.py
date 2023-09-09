from abc import ABC, abstractmethod
from functools import wraps


class Policy(ABC):
    @abstractmethod
    def act(self, obs, **kwargs):
        pass


def multi_agent_wrapper(policy_class):
    @wraps(policy_class)
    class MultiAgentPolicy(Policy):
        def __init__(self, num_agents, **kwargs):
            policy_class.__init__(self, **kwargs)
            self.num_agents = num_agents

        def act(self, obs, **kwargs):
            return [policy_class.act(self, obs[i], **kwargs) for i in range(self.num_agents)]
        
    return MultiAgentPolicy
