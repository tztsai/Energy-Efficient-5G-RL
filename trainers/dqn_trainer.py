import argparse
import os
import random
import time
from argparse import ArgumentParser

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from stable_baselines3.common.buffers import ReplayBuffer

from agents.dqn import QNetwork
from .trainer import BaseTrainer
from utils import notice


def get_dqn_config():
    parser = ArgumentParser()
    
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=20000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.03,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=20000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    
    return parser


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        self.observation_space = self.envs.observation_space[0]
        self.action_space = self.envs.action_space[0]

        args = config['all_args']
        self.lr = args.learning_rate
        self.q_net = QNetwork(self.observation_space, self.action_space).to(self.device)
        self.targ_net = QNetwork(self.observation_space, self.action_space).to(self.device)
        self.targ_net.load_state_dict(self.q_net.state_dict())

        self.rb = ReplayBuffer(
            args.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            handle_timeout_termination=False,
        )
    
    def train(self):
        optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, weight_decay=1e-5)
        start_time = time.time()

        envs = self.envs
        args = self.all_args
        writer = self.writer
        
        episodes = 0
        obs, _, _ = envs.reset()
        
        pbar = trange(args.num_env_steps // args.n_rollout_threads)
        for step in pbar:
            steps = (step + 1) * args.n_rollout_threads
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.num_env_steps, steps)
            
            if random.random() < epsilon:
                actions = np.array([[s.sample() for s in envs.action_space] for _ in range(envs.num_envs)])
            else:
                actions = self.take_actions(obs)
                
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, _ , rewards, done, infos, _ = envs.step(actions)

            obs = obs.reshape(-1, obs.shape[-1])
            obs1 = next_obs.reshape(-1, next_obs.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            rewards = np.repeat(rewards.reshape(-1, 1), self.num_agents, axis=1).reshape(-1)
            dones = np.repeat(done.reshape(-1, 1), self.num_agents, axis=1).reshape(-1)
            assert len(obs) == len(obs1) == len(actions) == len(rewards) == len(dones) \
                == envs.num_envs * self.num_agents
            
            # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
            for ob, ob1, a, r, d in zip(obs, obs1, actions, rewards, dones):
                self.rb.add(ob, ob1, a, r, d, {})

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            
            # ALGO LOGIC: training.
            
            if steps > args.learning_starts:
                if steps % args.train_frequency == 0:
                    data = self.rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.targ_net.net(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_net.gather(data.observations, data.actions)
                    loss = F.mse_loss(td_target, old_val)

                    if steps % 1000 == 0:
                        writer.add_scalar("losses/td_loss", loss, steps)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), steps)
                        writer.add_scalar("charts/SPS", int(steps / (time.time() - start_time)), steps)
                        pbar.set_postfix(SPS=int(steps / (time.time() - start_time)), td_loss=loss.item())

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if steps % args.target_network_frequency == 0:
                    for self.targ_net_param, self.net_param in zip(self.targ_net.parameters(), self.q_net.parameters()):
                        self.targ_net_param.data.copy_(
                            args.tau * self.net_param.data + (1.0 - args.tau) * self.targ_net_param.data
                        )
        
            if done.any():
                assert done.all()
                episodes += 1
                self.save()
                if episodes % 5 == 0:
                    self.save(version=f"_eps{episodes}")
                if episodes % self.log_interval == 0:
                    rew_df = pd.concat([pd.DataFrame(d['step_rewards']) for d in infos])
                    rew_info = rew_df.describe().loc[['mean', 'std', 'min', 'max']].unstack()
                    rew_info.index = ['_'.join(idx) for idx in rew_info.index]
                    self.log_train(rew_info, steps)

    def take_actions(self, obs):
        return np.array([self.q_net(torch.Tensor(ob).to(self.device))
                         .cpu().numpy() for ob in obs])

    def save(self, version=''):
        path = os.path.join(self.save_dir, f"dqn{version}.pt")
        notice(f"Saving model to {path}")
        torch.save(self.q_net.state_dict(), path)

    def load(self, version=''):
        path = os.path.join(self.model_dir, f"dqn{version}.pt")
        notice(f"Loading model from {path}")
        self.q_net.load_state_dict(torch.load(path))
        self.targ_net.load_state_dict(self.q_net.state_dict())
