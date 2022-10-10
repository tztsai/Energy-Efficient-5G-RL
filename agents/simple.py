import numpy as np
import pandas as pd
from gym.spaces import MultiDiscrete
from network.base_station import BaseStation as BS
from env.config import timeStep, actionInterval

class SimplePolicy:
    pre_sm2_steps = 5
    pre_sm3_steps = 25
    wakeup_threshold = 0.9

    def __init__(self, action_space, num_agents):
        self.act_space = action_space
        self.num_agents = num_agents
        self._sleep_steps = [0] * num_agents

    def act(self, obs, **__):
        def single_act(id, obs):
            info = BS.annotate_obs(obs)
            
            sm = info['sleep_mode']
            ants = info['num_antennas']
            next_sm = info['next_sleep_mode']
            wakeup_time = info['wakeup_time']
            thrp_req_queue = info['queued_sum_rate_req']
            thrp_req_idle = info['idle_sum_rate_req']
            arrival_rate = info['arrival_rate-1']
            thrp_req = info['serving_sum_rate_req']
            thrp = info['serving_sum_rate']
            
            nb_obs = info.iloc[next(i for i, k in enumerate(info.keys()) if k.startswith('nb0')):]
            nb_obs.index = pd.MultiIndex.from_tuples(
                list(map(tuple, nb_obs.index.str.split('_', 1))))
            nb_obs = nb_obs.unstack()
            rate_ratios = (nb_obs['sum_rate'] + 1) / (nb_obs['sum_rate_req'] + 1)

            new_sm = sm
            ant_switch = 1
            if sm:
                conn_mode = 1
                self._sleep_steps[id] += 1
                if sm != next_sm:
                    ant_switch = 2
                elif thrp_req_queue + thrp_req_idle or rate_ratios.min() < 0.8:  # wakeup
                    new_sm = 0
                    ant_switch = 2
                    if wakeup_time < 0.01:
                        conn_mode = 2
                elif sm == 1:
                    if self._sleep_steps[id] >= self.pre_sm2_steps:
                        new_sm = 2
                        ant_switch = 0
                # elif nb_obs.sum_rate.sum() < arrival_rate * 1.2:
                #     new_sm = 1
                #     ant_switch = 2
                elif sm == 2 and self._sleep_steps[id] >= self.pre_sm3_steps:
                    new_sm = 3
                    ant_switch = 0
                elif ants > 24:
                    ant_switch = 0
            else:
                conn_mode = 2
                self._sleep_steps[id] = 0
                if thrp_req == 0:
                    new_sm = 1
                elif thrp < thrp_req:
                    ant_switch = 2  # +4
                elif thrp > 1.2 * thrp_req:
                    ant_switch = 0  # -4
            return [ant_switch, new_sm, conn_mode]
        return list(map(single_act, range(self.num_agents), obs))
