import numpy as np
import pandas as pd
from gym.spaces import MultiDiscrete
from network.base_station import BaseStation as BS
from env.config import timeStep, actionInterval


class SimplePolicy:
    pre_sm2_steps = 10
    pre_sm3_steps = 50

    def __init__(self, action_space, num_agents):
        self.act_space = action_space
        self.num_agents = num_agents
        self._sleep_steps = [0] * num_agents

    def act(self, obs, **__):
        def single_act(id, obs):
            info = BS.annotate_obs(obs)
            sm = info['sleep_mode']
            next_sm = info['next_sleep_mode']
            wakeup_time = info['wakeup_time']
            # thrp_req_queue = info['queued_sum_rate_req']
            thrp_req_idle = info['idle_sum_rate_req']
            thrp_req = info['serving_sum_rate_req']
            thrp = info['serving_sum_rate']
            new_sm = sm
            ant_switch = 0
            if sm:
                conn_mode = 0
                self._sleep_steps[id] += 1
                # if sm != next_sm:
                #     pass
                if thrp_req_idle:  # wakeup
                    new_sm = 0
                    if wakeup_time < 5e-3:
                        conn_mode = 2
                elif sm == 1:
                    if self._sleep_steps[id] >= self.pre_sm2_steps:
                        new_sm = 2
                elif sm == 2 and self._sleep_steps[id] >= self.pre_sm3_steps:
                    new_sm = 3
            else:
                conn_mode = 2
                self._sleep_steps[id] = 0
                if thrp_req == 0:
                    new_sm = 1
                elif thrp / thrp_req > 2:
                    ant_switch = -1
                elif thrp / thrp_req < 1:
                    ant_switch = 1
            return [ant_switch + 1, new_sm, conn_mode]
        return list(map(single_act, range(self.num_agents), obs))


class SimplePolicySM1Only(SimplePolicy):
    pre_sm2_steps = 1e9
    pre_sm3_steps = 1e9


class SimplePolicyNoSM3(SimplePolicy):
    pre_sm3_steps = 1e9


class SleepyPolicy(SimplePolicy):
    pre_sm2_steps = 2
    pre_sm3_steps = 6
