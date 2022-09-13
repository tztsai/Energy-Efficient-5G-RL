import numpy as np
from gym.spaces import MultiDiscrete
from network.base_station import BaseStation as BS
from env.config import timeStep, actionInterval


class SleepyPolicy:
    """ Go to sleep whenever there is no traffic. """
    pre_sm2_time = 0.01
    mutual_obs_start = BS.public_obs_ndims + BS.private_obs_ndims
    neighbor_obs_dims = BS.public_obs_ndims + BS.mutual_obs_dim
    ue_stats_start = mutual_obs_start - BS.ue_stats_dim
    act_interval = timeStep * actionInterval
    wakeup_threshold = 0.

    def __init__(self, action_space, num_agents):
        self.act_space = action_space
        self.num_agents = num_agents
        self._sleep_timer = 0

    def act(self, obs):
        def single_act(obs):
            s = self.ue_stats_start
            sm = list(obs[2:6]).index(1)
            next_sm = list(obs[6:10]).index(1)
            wakeup_time = obs[10]
            arrival_rate = obs[s-1]
            thrp_cell = obs[s+4]
            thrp_log_ratio = obs[s+5]
            thrp_log_ratio_cell = obs[s+6]
            thrp_req = obs[s+7] + obs[s+8]
            thrp_req_idle = obs[s+9]
            obs_others = np.array(obs[self.mutual_obs_start:]).reshape(
                self.num_agents - 1, self.neighbor_obs_dims)
            
            new_sm = sm
            ant_switch = 2
            
            if sm:
                conn_mode = 1
                self._sleep_timer += self.act_interval
                if sm != next_sm:
                    pass
                # wakeup
                elif thrp_req_idle or obs_others[:, -1].min() < self.wakeup_threshold:
                    new_sm = 0
                    ant_switch = 4  # +16
                    if wakeup_time <= 3e-3:
                        conn_mode = 2
                elif sm == 1:
                    if self._sleep_timer >= self.pre_sm2_time:
                        new_sm = 2
                        ant_switch = 0  # -16
                elif sm > 1:
                    if obs_others[:, -2].sum() < arrival_rate * 1.2:
                        new_sm = 1
                        ant_switch = 4
            else:
                conn_mode = 2
                self._sleep_timer = 0
                if thrp_req == 0:
                    new_sm = 1
                    conn_mode = 0
                elif obs_others[:, -1].min() < 0:
                    ant_switch = 3  # +4
                elif thrp_log_ratio > 1:
                    ant_switch = 1  # -4
            return [ant_switch, new_sm, conn_mode]
        return [single_act(ob) for ob in obs]
