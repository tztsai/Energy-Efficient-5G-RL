import numpy as np
import pandas as pd
from .multi_agent import MultiAgentEnv
from .multi_discrete import MultiDiscrete
# from gym.spaces import MultiDiscrete
# from ray.rllib.env.multi_agent_env import MultiAgentEnv

from utils import info, debug, warn
from network.network import Green5GNet
from network.base_station import BaseStation
from network.config import areaSize, bsPositions
from visualize import render, animate
from . import config as C
from config import DEBUG


class Green5GNetEnv(MultiAgentEnv):
    """
    Action space:
    - Sleep mode: 0, 1, 2, 3
    - Connection mode: 0, 1, 2, 3
    - Switch antennae: -16, -4, 0, 4, 16
    """
    w_drop_cats = np.array(C.droppedAppWeights)
    w_drop = C.droppedTrafficWeight
    w_pc = C.powerConsumptionWeight
    episode_time_len = C.episodeTimeLen
    bs_poses = bsPositions
    num_agents = len(bsPositions)
    action_interval = C.actionInterval
    
    def __init__(self,
                 area_size=areaSize,
                 traffic_type=C.trafficType,
                 start_time=C.startTime,
                 time_step=C.timeStep,
                 accel_rate=C.accelRate,
                 action_interval=action_interval,
                 w_drop=w_drop,
                 w_pc=w_pc,
                 seed=0):
        super().__init__()
        
        self.net = Green5GNet(
            area=area_size,
            bs_poses=self.bs_poses,
            start_time=start_time,
            traffic_type=traffic_type,
            accel_rate=accel_rate)
        
        self.episode_len = int(self.episode_time_len / accel_rate /
                               time_step / action_interval)
        
        self.observation_space = [self.net.bs_obs_space
                                  for _ in range(self.num_agents)]
        self.cent_observation_space = self.net.net_obs_space
        self.action_space = [MultiDiscrete(BaseStation.action_dims)
                             for _ in range(self.num_agents)]
        
        self.w_drop = w_drop
        self.w_pc = w_pc
        self._reward_stats = []
        
        self._seed = seed
        self._dt = time_step
        self._episode_count = 0
        self._total_steps = 0

    def seed(self, seed=None):
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        
    @property
    def need_action(self):
        return self._sim_steps % self.action_interval == 0
        
    def get_reward(self):
        pc = self.net.power_consumption
        dr = self.net.drop_rates
        info('Power consumption: {}'.format(pc))
        info('Dropped rates: {}'.format(dr))
        dropped = np.sum(dr * self.w_drop_cats)
        return -self.w_drop * dropped - self.w_pc * pc

    def get_obs_agent(self, agent_id):
        return self.net.observe_bs(agent_id)

    def get_cent_obs(self):
        return [self.net.observe_network()]
    
    def reset(self):
        self.seed()
        self.net.reset()
        self._episode_steps = 0
        self._sim_steps = 0
        self._figure = None
        if DEBUG:
            self._steps_info = [self.net.info_dict()]   
        return self.get_obs(), self.get_cent_obs(), None
    
    def step(self, actions=None, substeps=action_interval):
        self.net.reset_stats()
        
        if actions is not None:
            for i in range(self.num_agents):
                self.net.set_action(i, actions[i])

        for i in range(substeps):
            self.net.step(self._dt)

        self._sim_steps += substeps
        self._total_steps += substeps / self.action_interval

        obs = self.get_obs()
        cent_obs = self.get_cent_obs()
        
        episode_steps = self._sim_steps // self.action_interval
        info(f'Step {episode_steps}:')
        
        reward = self.get_reward()
        info('Reward: %.2f', reward)

        if DEBUG:
            infos = self.net.info_dict()
            infos['reward'] = reward
            self._steps_info.append(infos)
            info('\nTime: %s', infos['time'])
            # info('\nBS states:\n{}'.format(info_dict['bs_info']))
            # info('\nUE states:\n{}'.format(info_dict['ue_info']))
            info('\nStatistics:')
            info('  %d users done', infos['total_done_count'])
            info('  %d users dropped', infos['total_dropped_count'])
            info('  data rate (Mbps): %.2f, %.2f, %.2f', *infos['avg_data_rates'])
            info('  drop rate (Mbps): %.2f, %.2f, %.2f', *infos['avg_drop_rates'])
            info('  drop ratio: %.2f%%, %.2f%%, %.2f%%', *infos['total_drop_ratios']*100)

        rewards = [[reward]]  # shared reward for all agents

        done = episode_steps >= self.episode_len
        if done:
            self._episode_count += 1

        return obs, cent_obs, rewards, done, {}, None

    def close(self):
        if DEBUG:
            pd.Series(self._steps_info, name='info').to_frame(
                ).to_feather('steps_info.feather')
    
    render = render
    animate = animate
