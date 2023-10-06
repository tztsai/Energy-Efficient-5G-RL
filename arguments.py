import argparse
from config import DEBUG, EVAL


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("-A", "--algorithm_name", type=str,
                        default='mappo', choices=["rmappo", "mappo", "dqn"])
    parser.add_argument("-G", "--group_name", type=str,
                        help="the group name of the training (we use traffic scenario)")
    parser.add_argument("-L", '--log_level', type=str, default='DEBUG' if DEBUG else 'NOTICE',
                        help='level of logging')
    parser.add_argument("-E", "--experiment_name", type=str, default="check", 
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("-T", "--num_env_steps", type=int, default=1000000,
                        help='Number of environment steps to train (default: 1000000)')
    parser.add_argument("--user_name", type=str, default='tcai7',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false',
                        help="[for wandb usage], will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='MultiCellNetwork', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, help="Max length for any episode")

    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1,
                        help="time duration between continuous twice log printing.")
    parser.add_argument("--sim_log_path",
                        help="path to save the log of the simulation")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=5, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=1, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("-R", "--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--use_dash", action='store_true', default=False, help="If set, use Dash to animate the network.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="by default None. set the path to pretrained model.")
    parser.add_argument("--run_version", default='',
                        help="any substring of the version name of the run")
    parser.add_argument("-V", "--model_version", type=str, default='',
                        help="by default None. set the version of pretrained model.")
    
    return parser


def get_env_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--scenario", type=str,
                        help="the traffic scenario of the simulation")
    parser.add_argument("-T", "--episode_len", type=int,
                        help="number of steps per episode")
    parser.add_argument("-T0", "--start_time", type=str,
                        help="start time of the simulation")
    parser.add_argument("-a", "--accelerate", type=int,
                        help="acceleration rate of the simulation")
    parser.add_argument("--area_size", type=float,
                        help="width of the square area in meters")
    parser.add_argument("--dpi_sample_rate", type=float,
                        help="DPI sample rate (inversely proportion to traffic density)")
    parser.add_argument("-s", "--save_trajectory", action='store_false',
                        help="save detailed steps info of the simulation")
    parser.add_argument("--stats_dir",
                        help="path to save steps info of the simulation")
    parser.add_argument("--include_bs_info", action='store_true')
    parser.add_argument("--no_interf", action='store_true')
    parser.add_argument("--no_offload", action='store_true')
    parser.add_argument("--max_sleep", type=int, default=3)
    # parser.add_argument("--w_pc", type=float,
    #                     help="weight of power consumption in reward")
    parser.add_argument("--w_qos", type=float,
                        help="weight of QoS in reward")
    parser.add_argument("--w_xqos", type=float,
                        help="weight of extra QoS in QoS reward"),
    return parser
