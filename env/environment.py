import numpy as np
import pandas as pd
from .multi_agent import MultiAgentEnv
from .multi_discrete import MultiDiscrete
# from gym.spaces import MultiDiscrete
# from ray.rllib.env.multi_agent_env import MultiAgentEnv

from . import config
from utils import *
from config import *
from network.network import MultiCellNetwork
from network.base_station import BaseStation
from traffic import TrafficModel
from network import config as net_config
from visualize import render, animate


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

    def __init__(
        self,
        area_size=net_config.areaSize,
        scenario=config.trafficScenario,
        start_time=config.startTime,
        episode_len=None,
        time_step=config.timeStep,
        accelerate=config.accelRate,
        action_interval=action_interval,
        dpi_sample_rate=None,
        w_drop=w_drop,
        w_pc=w_pc,
        w_delay=w_delay,
        seed=None,
        save_trajectory=False,
        include_bs_info=False,
        stats_dir=None,
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
        
        if episode_len is None:
            episode_len = round(self.episode_time_len / accelerate / time_step / action_interval)
        self.episode_len = episode_len
        self.action_interval = action_interval
        
        self.observation_space = [self.net.bs_obs_space
                                  for _ in range(self.num_agents)]
        self.cent_observation_space = self.net.net_obs_space
        self.action_space = [MultiDiscrete(BaseStation.action_dims)
                             for _ in range(self.num_agents)]

        self.w_drop = w_drop
        self.w_pc = w_pc
        self.w_delay = w_delay
        self.stats_dir = stats_dir
        self.save_trajectory = save_trajectory
        self.include_bs_info = include_bs_info
        
        self._seed = seed
        self._dt = time_step
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        
        self.seed()

    def print_info(self):
        notice('Start time: {}'.format(self.net.world_time_repr))
        notice('Acceleration: {}'.format(self.net.accelerate))
        notice('Time step: {} ms <-> {} s'
               .format(self._dt * 1000, self._dt * self.net.accelerate))
        notice('Action interval: {} ms <-> {} min'
               .format(self.action_interval * self._dt * 1000,
                       self.action_interval * self._dt * self.net.accelerate / 60))
        notice('Episode length: {}'.format(self.episode_len))
        notice('Episode time length: {} h'.format(self.episode_time_len / 3600))
        notice('Power consumption weight: {}'.format(self.w_pc))
        notice('Drop ratio weight: {}'.format(self.w_drop))
        notice('Delay weight: {}'.format(self.w_delay))
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
        TrafficModel.seed(seed)

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
        reward = -(self.w_drop * dropped + self.w_pc * pc + self.w_delay * delay)
        r_info = dict(drop=dropped, delay=delay, pc=pc, reward=reward)
        if EVAL:
            # r_info['drop_counts'] = self.net._eval_stats['num_dropped'].values.copy()
            r_info['drop_ratios'] = dr
            r_info['ue_delays'] = dl
        self._reward_stats.append(r_info)
        return reward

    def get_obs_agent(self, agent_id):
        return self.net.observe_bs(agent_id)

    def get_cent_obs(self):
        return [self.net.observe_network()]
    
    def reset(self, render_mode=None):
        # self.seed()
        self.net.reset()
        self._episode_steps = 0
        self._sim_steps = 0
        self._figure = None
        self._reward_stats = []
        if EVAL and self.save_trajectory:
            self.net._other_stats['reward'] = self._reward_stats
            self._trajectory = [self.info_dict()]
        self.render(render_mode)
        return self.get_obs(), self.get_cent_obs(), None
    
    def step(self, actions=None, substeps=action_interval, 
             render_mode=None, render_interval=1):
        if EVAL:
            info(f'\nStep {self._sim_steps}:\n')
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
            if i == substeps - 1:  # update stats before rendering
                self.net.update_stats()
            if render_mode and (i + 1) % render_interval == 0:
                self.render(render_mode)

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

        if EVAL:
            info('')
            infos = self.info_dict()
            for k, v in infos.items():
                if k.startswith('bs_'): continue
                info('%s: %s', k, v)
            if self.save_trajectory:
                self._trajectory.append(infos)

        if done:
            self._episode_count += 1
            if TRAIN:  # only for training logging
                infos['step_rewards'] = self._reward_stats
                infos['sm3_ratio'] = self.net.avg_sleep_ratios()[3]

            notice('Episode %d finished at %s', self._episode_count, self.net.world_time_repr)
        
        return obs, cent_obs, rewards, done, infos, None
    
    def info_dict(self):
        info = self.net.info_dict(include_bs=self.include_bs_info)
        info.update(
            reward = self._sim_steps and self._reward_stats[-1]['reward'],
            weighted_pc = self._sim_steps and self._reward_stats[-1]['pc'] * self.w_pc,
            weighted_drop = self._sim_steps and self._reward_stats[-1]['drop'] * self.w_drop,
            weighted_delay = self._sim_steps and self._reward_stats[-1]['delay'] * self.w_delay,
        )
        return info

    def close(self):
        if EVAL:
            stats_dir = os.path.join(self.stats_dir, self.net.traffic_model.scenario.name)
            os.makedirs(stats_dir, exist_ok=True)
            self.net.save_stats(stats_dir)
            if self.save_trajectory:
                path = os.path.join(stats_dir, 'trajectory.csv')
                pd.DataFrame(self._trajectory).set_index('time').to_csv(path)
    
    render = render
    animate = animate
