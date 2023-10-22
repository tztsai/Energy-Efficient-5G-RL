import os
import numpy as np
import torch
import wandb
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(ABC):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.num_agents = config['num_agents']
        self.device = config['device']
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            if wandb.run:
                self.save_dir = self.run_dir = str(wandb.run.dir)
            else:
                print("Wandb is not initialized. Disabling wandb.")
        else:
            self.run_dir = config["run_dir"]
            self.save_dir = str(self.run_dir / 'models')
            self.log_dir = str(self.run_dir / 'logs')
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)

    @abstractmethod
    def train(self):
        """Collect training data, perform training updates, and evaluate policy."""
        
    @abstractmethod
    def take_actions(self, obs):
        """Use the policy to take actions in the environment."""
    
    @abstractmethod
    def save(self, version=''):
        """Save policy's actor and critic networks."""

    @abstractmethod
    def load(self, version=''):
        """Load policy's networks from a saved model."""
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writer.add_scalars(k, {k: v}, total_num_steps)

    def close(self):
        self.envs.close()
        if self.eval_envs:
            self.eval_envs.close()
        
        if self.use_wandb:
            wandb.run.finish()
        else:
            # self.writer.export_scalars_to_json(
            #     str(self.log_dir + '/summary.json'))
            self.writer.close()
        