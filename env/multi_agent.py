from gymnasium import Env
from functools import lru_cache


class MultiAgentEnv(Env):
    num_agents: int
    # _obs_space: spaces.Space
    # _cent_obs_space: spaces.Space
    # _act_space: spaces.Space

    # @property
    # def _agent_ids(self):
    #     return set(range(self.num_agents))
    
    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.num_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        raise NotImplementedError

    def get_cent_obs(self):
        """Returns the global state."""
        raise NotImplementedError

    def get_cent_obs_size(self):
        """Returns the size of the global state."""
        raise NotImplementedError

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.num_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        raise NotImplementedError

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        return {
            "cent_obs_shape": self.get_cent_obs_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.num_agents
        }
