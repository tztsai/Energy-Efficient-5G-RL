import numpy as np
import pandas as pd
from .multi_agent import MultiAgentEnv
from .multi_discrete import MultiDiscrete
# from gym.spaces import MultiDiscrete
# from ray.rllib.env.multi_agent_env import MultiAgentEnv

from utils import info, debug, warn, notice
from network.network import MultiCellNetwork
from network.base_station import BaseStation
from network.config import areaSize, bsPositions
from visualize import render, animate
from . import config as C
from config import DEBUG


class MultiCellNetEnv(MultiAgentEnv):
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
        
        self.net = MultiCellNetwork(
            area=area_size,
            bs_poses=self.bs_poses,
            start_time=start_time,
            traffic_type=traffic_type,
            accel_rate=accel_rate)
        
        self.episode_len = int(self.episode_time_len / accel_rate /
                               time_step / action_interval)
        self.action_interval = action_interval
        
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
        self._episode_steps = 0
        self._total_steps = 0
        
    def print_info(self):
        notice('Traffic type: {}'.format(self.net.traffic_model.scenario))
        notice('Start time: {} s'.format(self.net.start_time))
        notice('Time step: {} ms'.format(self._dt * 1000))
        notice('Acceleration: {}'.format(self.net.accel_rate))
        notice('Action interval: {} ms'.format(self.action_interval * self._dt * 1000))
        notice('Episode length: {}'.format(self.episode_len))
        notice('Episode time length: {} h'.format(self.episode_time_len / 3600))
        notice('Drop rate weight: {}'.format(self.w_drop))
        notice('Power consumption weight: {}'.format(self.w_pc))
        notice('Observation space: {}'.format(
            (self.num_agents, *self.observation_space[0].shape)))
        notice('Central observation space: {}'.format(
            self.cent_observation_space.shape))
        notice('Action space: {}'.format(
            (self.num_agents, self.action_space[0].shape)))

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
        if DEBUG and not hasattr(self, '_steps_info'):
            self._steps_info = [self.info_dict()]   
        return self.get_obs(), self.get_cent_obs(), None
    
    def step(self, actions=None, substeps=action_interval):
        info(f'\nStep {self._sim_steps}:\n')
        
        self.net.reset_stats()
        
        if actions is not None:
            for i in range(self.num_agents):
                self.net.set_action(i, actions[i])

        for i in range(substeps):
            self.net.step(self._dt)

        self.net.update_stats()

        steps = substeps / self.action_interval
        self._sim_steps += substeps
        self._total_steps += steps
        self._episode_steps += steps

        obs = self.get_obs()
        cent_obs = self.get_cent_obs()
        reward = self.get_reward()

        if DEBUG:
            infos = self.info_dict()
            self._steps_info.append(infos)
            info('\nTime: %s', infos['time'])
            info('\nBS states:\n{}'.format(infos['bs_info']))
            # ue_info = infos.pop('ue_info')
            # info('\nUE states:\n{}'.format(info_dict['ue_info']))
            info('\nStatistics:')
            info('  %d users done', infos['total_done_count'])
            info('  %d users dropped', infos['total_dropped_count'])
            info('  data rate (Mbps): %.2f, %.2f, %.2f', *infos['avg_data_rates'])
            info('  drop rate (Mbps): %.2f, %.2f, %.2f', *infos['avg_drop_rates'])
            info('  drop ratio: %.2f%%, %.2f%%, %.2f%%', *infos['total_drop_ratios']*100)

        info('Reward: %.2f', reward)

        rewards = [[reward]]  # shared reward for all agents

        done = self._episode_steps >= self.episode_len
        if done:
            self._episode_count += 1

        return obs, cent_obs, rewards, done, {}, None
    
    def info_dict(self):
        info = self.net.info_dict()
        info['reward'] = self.get_reward()
        return info

    def close(self):
        if DEBUG:
            bs_df = pd.concat([info.pop('bs_info') for info in self._steps_info],
                              keys=range(len(self._steps_info))).unstack()
            bs_df.columns = bs_df.columns.map(lambda p: f'bs_{p[1]}_{p[0]}')
            df = pd.concat([pd.DataFrame(self._steps_info), bs_df], axis=1)
            df.set_index('time').to_csv('results/steps_info.csv')
    
    render = render
    animate = animate
