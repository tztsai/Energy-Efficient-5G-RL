import numpy as np
import pandas as pd
from .multi_agent import MultiAgentEnv
from .multi_discrete import MultiDiscrete
# from gym.spaces import MultiDiscrete
# from ray.rllib.env.multi_agent_env import MultiAgentEnv

from . import config
from utils import info, debug, warn, notice
from network.network import MultiCellNetwork
from network.base_station import BaseStation
from network import config as net_config
from visualize import render, animate
from config import *


class MultiCellNetEnv(MultiAgentEnv):
    """
    Action space:
    - Sleep mode: 0, 1, 2, 3
    - Connection mode: 0, 1, 2, 3
    - Switch antennae: -16, -4, 0, 4, 16
    """
    w_drop_cats = np.array(config.droppedAppWeights)
    w_delay_cats = np.array(config.delayAppWeights)
    w_drop = config.droppedTrafficWeight
    w_delay = config.delayWeight
    w_pc = config.powerConsumptionWeight
    episode_time_len = config.episodeTimeLen
    bs_poses = net_config.bsPositions
    num_agents = len(bs_poses)
    action_interval = config.actionInterval
    
    def __init__(self,
                 area_size=net_config.areaSize,
                 traffic_type=config.trafficType,
                 start_time=config.startTime,
                 time_step=config.timeStep,
                 accel_rate=config.accelRate,
                 action_interval=action_interval,
                 w_drop=w_drop,
                 w_pc=w_pc,
                 w_delay=w_delay,
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
        self.w_delay = w_delay
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
    
    def get_reward(self, obs=None):
        if obs is None:
            pc = self.net.power_consumption
            dr = self.net.drop_rates
            dl = self.net.service_delays
        else:  # use already calculated values
            pc = obs[0]
            dr = obs[1:4]
            dl = obs[4:7]
            if DEBUG:
                assert abs(pc - self.net.power_consumption) < 1e-6
                assert np.abs(dr - self.net.drop_rates).sum() < 1e-4
                assert np.abs(dl - self.net.service_delays).sum() < 1e-4
        dropped = dr @ self.w_drop_cats
        delay = dl @ self.w_delay_cats
        return -(self.w_drop * dropped + self.w_pc * pc + self.w_delay * delay)

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
        if EVAL and not hasattr(self, '_steps_info'):
            self._steps_info = [self.info_dict()]   
        return self.get_obs(), self.get_cent_obs(), None
    
    def step(self, actions=None, substeps=action_interval):
        if EVAL:
            notice(f'\nStep {self._sim_steps}:\n')
            info('traffic distribution: %s',
                 self.net.traffic_model.get_arrival_rates(self.net.world_time, self._dt))
            
        self.net.reset_stats()
        
        if actions is not None:
            for i in range(self.num_agents):
                self.net.set_action(i, actions[i])

        for i in range(substeps):
            if EVAL:
                debug('Substep %d', i + 1)
            self.net.step(self._dt)

        self.net.update_stats()

        steps = substeps / self.action_interval
        self._sim_steps += substeps
        self._total_steps += steps
        self._episode_steps += steps

        obs = self.get_obs()
        cent_obs = self.get_cent_obs()
        reward = self.get_reward(cent_obs[0])

        rewards = [[reward]]  # shared reward for all agents

        done = self._episode_steps >= self.episode_len
        infos = {}

        if done:
            self._episode_count += 1

        if EVAL:
            notice('Reward: %.2f', reward)

        if EVAL and self._episode_steps % 4 == 0:
            infos = self.info_dict()
            self._steps_info.append(infos)
            notice('\nTime: %s', infos['time'])
            notice('Power consumption: {}'.format(self.net.power_consumption))
            notice('Arrival rates: {}'.format(self.net.arrival_rates))
            notice('Dropped rates: {}'.format(self.net.drop_rates))
            notice('Service delays: {}'.format(self.net.service_delays))
            # info('\nBS states:\n{}'.format(infos['bs_info']))
            # info('\nUE states:\n{}'.format(infos['ue_info']))
            notice('\nStatistics:')
            notice('  average PC: %.3f', infos['avg_pc']),
            notice('  %d users done', infos['total_done_count'])
            notice('  %d users dropped', infos['total_dropped_count'])
            notice('  done traffic (Mb): %.2f, %.2f, %.2f', *infos['total_done_vol']),
            notice('  dropped traffic (Mb): %.2f, %.2f, %.2f', *infos['total_dropped_vol']),
            notice('  latency (ms): %.1f, %.1f, %.1f', *infos['avg_latency'])
            notice('  data rate (Mbps): %.2f, %.2f, %.2f', *infos['avg_data_rates'])
            notice('  drop rate (Mbps): %.2f, %.2f, %.2f', *infos['avg_drop_rates'])
            notice('  drop ratio: %.2f%%, %.2f%%, %.2f%%', *infos['total_drop_ratios']*100)

        return obs, cent_obs, rewards, done, infos, None
    
    def info_dict(self):
        info = self.net.info_dict()
        info['reward'] = self.get_reward()
        return info

    def close(self):
        if EVAL:
            bs_df = pd.concat([info.pop('bs_info') for info in self._steps_info],
                              keys=range(len(self._steps_info))).unstack()
            bs_df.columns = bs_df.columns.map(lambda p: f'bs_{p[1]}_{p[0]}')
            df = pd.concat([pd.DataFrame(self._steps_info), bs_df], axis=1)
            df.set_index('time').to_csv('results/steps_info.csv')
    
    render = render
    animate = animate
