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

from utils import set_log_level
from env.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def get_env_kwargs(args):
    return {k: v for k, v in vars(args).items() if v is not None}


def make_env(args, env_args, for_eval=False):
    n_threads = args.n_rollout_threads
    if args.episode_length is None:
        tmp_env = MultiCellNetEnv(**get_env_kwargs(env_args))
        tmp_env.print_info()
        args.episode_length = tmp_env.episode_len // n_threads

    def get_env_fn(rank):
        def init_env():
            kwargs = get_env_kwargs(env_args)
            kwargs.setdefault('start_time',
                              rank / n_threads * MultiCellNetEnv.episode_time_len)
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
    parser.add_argument("--traffic_type", type=str, help="type of traffic to generate")
    parser.add_argument("--start_time", type=str, help="start time of the simulation")
    parser.add_argument("--accel_rate", type=float, help="acceleration rate of the simulation")
    parser.add_argument("--w_drop", type=float, help="weight of dropped rate in reward")
    parser.add_argument("--w_pc", type=float, help="weight of power consumption in reward")

    return parser.parse_args(args)


def main(args):
    parser = get_config()
    args, env_args = parser.parse_known_args(args)
    env_args = parse_env_args(env_args)
    
    if args.algorithm_name == "rmappo":
        assert (args.use_recurrent_policy or args.use_naive_recurrent_policy), (
            "check recurrent policy!")
    elif args.algorithm_name == "mappo":
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    # cuda
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" \
        / args.env_name / env_args.traffic_type / args.algorithm_name / args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if args.use_wandb:
        run = wandb.init(config=args,
                         project=args.env_name,
                         entity=args.user_name,
                         notes=socket.gethostname(),
                         name=str(args.algorithm_name) + "_" +
                         str(args.experiment_name) +
                         "_seed" + str(args.seed),
                         group=env_args.traffic_type,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # setproctitle.setproctitle(str(args.algorithm_name) + "-" +
    #                           str(args.env_name) + "-" + str(args.experiment_name) + "@" + str(args.user_name))

    set_log_level(args.log_level)
    
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # env init
    envs = make_env(args, env_args)
    eval_envs = make_env(args, env_args, for_eval=True) if args.use_eval else None

    config = {
        "all_args": args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": MultiCellNetEnv.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    # if args.share_policy:
    #     from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    # else:
    #     from onpolicy.runner.separated.mpe_runner import MPERunner as Runner
    from runner import Green5GRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
