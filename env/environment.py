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
    - Switch antennas: -16, -4, 0, 4, 16
    """
    w_drop_cats = np.array(config.dropAppWeights)
    w_delay_cats = np.array(config.delayAppWeights)
    w_drop = config.dropRatioWeight
    w_delay = config.delayWeight
    w_pc = config.powerConsumptionWeight
    episode_time_len = config.episodeTimeLen
    bs_poses = net_config.bsPositions
    num_agents = len(bs_poses)
    action_interval = config.actionInterval
    stats_save_path = 'analysis/stats.csv'
    
    def __init__(
        self,
        area_size=net_config.areaSize,
        scenario=config.trafficScenario,
        start_time=config.startTime,
        time_step=config.timeStep,
        accelerate=config.accelRate,
        action_interval=action_interval,
        dpi_sample_rate=None,
        w_drop=w_drop,
        w_pc=w_pc,
        w_delay=w_delay,
        seed=0,
        stats_save_path=stats_save_path,
    ):
        super().__init__()
        
        self.net = MultiCellNetwork(
            area=area_size,
            bs_poses=self.bs_poses,
            start_time=start_time,
            traffic_scenario=scenario,
            accelerate=accelerate,
            dpi_sample_rate=dpi_sample_rate
        )
        
        self.episode_len = int(self.episode_time_len / accelerate /
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
        self.stats_save_path = stats_save_path
        
        self._seed = seed
        self._dt = time_step
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        
    def print_info(self):
        notice('Start time: {} s'.format(self.net.start_time))
        notice('Time step: {} ms'.format(self._dt * 1000))
        notice('Acceleration: {}'.format(self.net.accelerate))
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
            dr = self.net.drop_ratios
            dl = self.net.service_delays
        else:  # use already calculated values
            pc = obs[0]
            dr = obs[1:4]
            dl = obs[4:7]
            if DEBUG:
                assert abs(pc - self.net.power_consumption) < 1e-3
                assert np.abs(dr - self.net.drop_ratios).sum() < 1e-4
                assert np.abs(dl - self.net.service_delays).sum() < 1e-5
        dropped = dr @ self.w_drop_cats
        delay = dl @ self.w_delay_cats
        penalty = self.w_drop * dropped + self.w_pc * pc + self.w_delay * delay
        if EVAL:
            self.net.add_stat('reward', dict(
                n_drop=self.net._eval_stats['num_dropped'].values.copy(),
                v_drop=dr, drop=dropped,
                v_delay=dl, delay=delay,
                pc=pc, penalty=penalty))
        return -penalty

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
    
    def step(self, actions=None, substeps=action_interval, 
             render_interval=None, render_mode=None):
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
            if render_interval is not None and (i + 1) % render_interval == 0:
                self.render(render_mode)

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
            notice('')
            infos = self.info_dict()
            self._steps_info.append(infos)
            for k, v in infos.items():
                if k.startswith('bs_'): continue
                notice('%s: %s', k, v)

        return obs, cent_obs, rewards, done, infos, None
    
    def info_dict(self):
        info = self.net.info_dict()
        info['reward'] = self.get_reward()
        return info

    def close(self):
        if EVAL:
            pd.DataFrame(self._steps_info).set_index('time').to_csv(self.stats_save_path)
            self.net.save_other_stats()
    
    render = render
    animate = animate
