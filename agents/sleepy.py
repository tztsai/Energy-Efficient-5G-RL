import numpy as np
from gym.spaces import MultiDiscrete
from network.base_station import BaseStation as BS
from env.config import timeStep, actionInterval

class SleepyPolicy:
    pre_sm2_time = 0.01
    pre_sm3_time = 0.05
    mutual_obs_start = BS.public_obs_dim + BS.private_obs_dim
    neighbor_obs_dims = BS.public_obs_dim + BS.mutual_obs_dim
    act_interval = timeStep * actionInterval
    wakeup_threshold = 0.1

    def __init__(self, action_space, num_agents):
        self.act_space = action_space
        self.num_agents = num_agents
        self._sleep_timer = 0

    def act(self, obs, **__):
        def single_act(obs):
            # s = self.ue_stats_start
            # sm = list(obs[2:6]).index(1)
            # next_sm = list(obs[6:10]).index(1)
            # wakeup_time = obs[10]
            # arrival_rate = obs[s-1]
            # thrp_cell = obs[s+4]
            # thrp_log_ratio = obs[s+5]
            # thrp_log_ratio_cell = obs[s+6]
            # thrp_req = obs[s+7] + obs[s+8]
            # thrp_req_idle = obs[s+9]
            obs_others = obs[self.mutual_obs_start:].reshape(
                self.num_agents - 1, self.neighbor_obs_dims)
            info = BS.annotate_obs(obs)
            
            sm = info['sleep_mode']
            next_sm = info['next_sleep_mode']
            wakeup_time = info['wakeup_time']
            thrp_req_queue = info['thrp_req_queued']
            thrp_req_idle = info['thrp_req_idle']
            arrival_rate = info['arrival_rate-1']
            thrp_req = info['thrp_req_serving']
            log_ratio = info['log_ratio_serving']

            new_sm = sm
            ant_switch = 2
            if sm:
                conn_mode = 1
                self._sleep_timer += self.act_interval
                if sm != next_sm:
                    pass
                elif thrp_req_queue + thrp_req_idle:  # wakeup
                    new_sm = 0
                    ant_switch = 2  # increase
                    if wakeup_time <= 3e-3:
                        conn_mode = 2
                elif sm == 1:
                    if self._sleep_timer >= self.pre_sm2_time:
                        new_sm = 2
                        ant_switch = 0  # decrease
                else:
                    if obs_others[:,-2].sum() < arrival_rate * 1.2:
                        new_sm = 1
                        ant_switch = 2
                    elif sm == 2:
                        if self._sleep_timer >= self.pre_sm3_time:
                            new_sm = 3
                            ant_switch = 0
            else:
                conn_mode = 2
                self._sleep_timer = 0
                if thrp_req == 0:
                    new_sm = 1
                    ant_switch = 0
                elif log_ratio < 0:
                    ant_switch = 2 
                elif log_ratio > 1:
                    ant_switch = 0 
            return [ant_switch, new_sm, conn_mode]
        return [single_act(ob) for ob in obs]
