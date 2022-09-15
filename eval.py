#!/usr/bin/env python
import sys
import os
import wandb
import socket
# import setproctitle
import numpy as np
from pathlib import Path
import torch
from arguments import get_config
from env import MultiCellNetEnv
from argparse import ArgumentParser

from utils import logger
from env.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def get_env_kwargs(args):
    return {k: v for k, v in vars(args).items() if v is not None}


def make_env(args, env_args, for_eval=False):
    n_threads = args.n_rollout_threads
    if args.episode_length is None:
        tmp_env = MultiCellNetEnv(**get_env_kwargs(env_args))
        args.episode_length = tmp_env.episode_len // n_threads
        print("Episode length: {}".format(args.episode_length))

    def get_env_fn(rank):
        def init_env():
            kwargs = get_env_kwargs(env_args)
            kwargs.setdefault('start_time', rank / n_threads * MultiCellNetEnv.episode_time_len)
            env = MultiCellNetEnv(**kwargs)
            if for_eval:
                env.seed(args.seed * 50000 + rank * 10000)
            else:
                env.seed(args.seed + rank * 1000)
            return env
        return init_env
    
    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def parse_env_args(args):
    parser = ArgumentParser()
    
    parser.add_argument("--area_size", type=float, help="width of the square area in meters")
    parser.add_argument("--traffic_type", default="urban", type=str, help="type of traffic to generate")
    parser.add_argument("--start_time", type=str, help="start time of the simulation")
    parser.add_argument("--accelerate", type=float, help="acceleration rate of the simulation")
    parser.add_argument("--act_interval", type=int, help="number of simulation steps between two actions")

    return parser.parse_args(args)


def get_latest_model_dir(run_dir, use_wandb=True):
    if use_wandb:
        pat = 'wandb/run*/files'
    else:
        pat = 'run*/models'
    return max(run_dir.glob(pat), key=os.path.getmtime)
    

def main(args):
    parser = get_config()
    args, env_args = parser.parse_known_args(args)
    env_args = parse_env_args(env_args)
    
    if args.algorithm_name == "rmappo":
        assert (args.use_recurrent_policy or args.use_naive_recurrent_policy), (
            "check recurrent policy!")
    elif args.algorithm_name == "mappo":
        assert (args.use_recurrent_policy ==
                False and args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    device = torch.device("cpu")

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
        / args.env_name / env_args.traffic_type / args.algorithm_name / args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        
    args.model_dir = get_latest_model_dir(run_dir, use_wandb=1)
    print('Model path:', args.model_dir)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # env init
    eval_envs = make_env(args, env_args, for_eval=True)

    config = {
        "all_args": args,
        "envs": eval_envs,
        "eval_envs": eval_envs,
        "num_agents": MultiCellNetEnv.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    # if args.share_policy:
    #     from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    # else:
    #     from onpolicy.runner.separated.mpe_runner import MPERunner as Runner
    from runner import MultiCellNetRunner as Runner

    runner = Runner(config)
    runner.restore()
    runner.eval(args.num_env_steps)

    # post process
    eval_envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
